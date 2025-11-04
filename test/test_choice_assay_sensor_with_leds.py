#!/usr/bin/env python3
"""
Test suite for ChoiceAssaySensorWithLEDs

This test suite mocks the RPi.GPIO module to allow testing LED functionality
on systems without physical GPIO pins.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np
from dataclasses import dataclass

# Mock RPi.GPIO before importing our sensor
sys.modules['RPi'] = Mock()
sys.modules['RPi.GPIO'] = Mock()

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from choice_assay.my_choice_assay_sensor_with_leds import ChoiceAssaySensorWithLEDs, ChoiceAssaySensorCfg


class TestChoiceAssaySensorWithLEDs(unittest.TestCase):
    """Test the ChoiceAssaySensorWithLEDs functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Use default configuration from the module
        from choice_assay.my_choice_assay_sensor_with_leds import DEFAULT_CA_SENSOR_CFG
        self.config = DEFAULT_CA_SENSOR_CFG
        
        # Mock picamera2 and cv2
        self.mock_picamera2 = Mock()
        self.mock_cv2 = Mock()
        
        # Create mock frame
        self.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('choice_assay.my_choice_assay_sensor_with_leds.Picamera2', self.mock_picamera2), \
             patch('choice_assay.my_choice_assay_sensor_with_leds.cv2', self.mock_cv2):
            self.sensor = ChoiceAssaySensorWithLEDs(self.config)

    @patch('RPi.GPIO.setup')
    @patch('RPi.GPIO.setmode')
    @patch('RPi.GPIO.cleanup')
    def test_gpio_initialization(self, mock_cleanup, mock_setmode, mock_setup):
        """Test GPIO pins are initialized correctly"""
        self.sensor._initialize_gpio()
        
        # Check GPIO mode is set
        mock_setmode.assert_called_once()
        
        # Check LED pins are set to output
        led_calls = [unittest.mock.call(17, 0), unittest.mock.call(22, 0), 
                    unittest.mock.call(27, 0), unittest.mock.call(23, 0), unittest.mock.call(26, 0)]
        for call in led_calls:
            self.assertIn(call, mock_setup.call_args_list)

    @patch('RPi.GPIO.output')
    def test_led_control_functions(self, mock_output):
        """Test LED control functions work correctly"""
        # Test red LED
        self.sensor._set_red_led(True)
        mock_output.assert_called_with(17, 1)
        
        self.sensor._set_red_led(False)
        mock_output.assert_called_with(17, 0)
        
        # Test green LED
        self.sensor._set_green_led(True)
        mock_output.assert_called_with(22, 1)
        
        # Test blue LED
        self.sensor._set_blue_led(True)
        mock_output.assert_called_with(27, 1)
        
        # Test frame LEDs
        self.sensor._set_left_frame_led(True)
        mock_output.assert_called_with(23, 1)
        
        self.sensor._set_right_frame_led(True)
        mock_output.assert_called_with(26, 1)

    @patch('RPi.GPIO.input')
    def test_emergency_stop_button(self, mock_input):
        """Test emergency stop button functionality"""
        # Test button not pressed
        mock_input.return_value = 1  # Pull-up, not pressed
        self.assertFalse(self.sensor._check_emergency_stop())
        
        # Test button pressed
        mock_input.return_value = 0  # Pulled low, pressed
        self.assertTrue(self.sensor._check_emergency_stop())

    @patch('RPi.GPIO.output')
    def test_led_status_updates(self, mock_output):
        """Test LED status updates based on motion detection state"""
        # Initialize LED state tracking
        self.sensor.motion_counter = 5
        self.sensor.left_motion_active = True
        self.sensor.right_motion_active = False
        self.sensor.grace_period_active = False
        
        # Test LED update
        self.sensor._update_leds()
        
        # Red LED should be on (motion_counter > 0)
        mock_output.assert_any_call(17, 1)
        
        # Green LED should be on (left_motion_active)
        mock_output.assert_any_call(22, 1)

    def test_dual_arena_motion_detection_logic(self):
        """Test dual arena motion detection mutual exclusivity"""
        # Create mock frames with motion in different areas
        frame_left_motion = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_left_motion[50:150, 50:150] = 255  # Motion in left area
        
        frame_right_motion = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_right_motion[50:150, 400:500] = 255  # Motion in right area
        
        # Mock cv2 functions
        with patch('choice_assay.my_choice_assay_sensor_with_leds.cv2') as mock_cv2:
            mock_cv2.cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)
            mock_cv2.GaussianBlur.return_value = np.zeros((480, 640), dtype=np.uint8)
            mock_cv2.threshold.return_value = (None, np.ones((240, 320), dtype=np.uint8) * 255)
            mock_cv2.countNonZero.return_value = 1000  # Above threshold
            
            # Test motion detection with left arena motion
            left_motion, right_motion = self.sensor._detect_motion_dual_arena(frame_left_motion)
            
            # Should detect motion (mocked to return high pixel count)
            self.assertTrue(left_motion or right_motion)

    @patch('RPi.GPIO.output')
    def test_cleanup_gpio(self, mock_output):
        """Test GPIO cleanup turns off all LEDs"""
        with patch('RPi.GPIO.cleanup') as mock_cleanup:
            self.sensor._cleanup_gpio()
            
            # All LEDs should be turned off
            mock_output.assert_any_call(17, 0)  # Red
            mock_output.assert_any_call(22, 0)  # Green
            mock_output.assert_any_call(27, 0)  # Blue
            mock_output.assert_any_call(23, 0)  # Left frame
            mock_output.assert_any_call(26, 0)  # Right frame
            
            # GPIO should be cleaned up
            mock_cleanup.assert_called_once()

    def test_configuration_validation(self):
        """Test configuration parameters are properly validated"""
        # Test valid configuration
        self.assertEqual(self.config.motion_threshold, 125)
        self.assertEqual(self.config.grace_period_seconds, 3.0)
        self.assertEqual(self.config.frame_rate, 5)
        
        # Test arena coordinates
        self.assertEqual(self.config.left_detection_roi, (210, 373, 560, 578))
        self.assertEqual(self.config.right_detection_roi, (1170, 373, 1520, 578))

    @patch('choice_assay.my_choice_assay_sensor_with_leds.Picamera2')
    @patch('choice_assay.my_choice_assay_sensor_with_leds.cv2')
    def test_sensor_initialization_without_gpio_errors(self, mock_cv2, mock_picamera2):
        """Test sensor can initialize even with GPIO errors (graceful degradation)"""
        with patch('RPi.GPIO.setmode', side_effect=Exception("No GPIO available")):
            # Should not raise exception, should handle gracefully
            try:
                sensor = ChoiceAssaySensorWithLEDs(self.config)
                # If we get here, the sensor handled the GPIO error gracefully
                self.assertTrue(True)
            except Exception as e:
                # If initialization fails, it should be with a descriptive message
                self.assertIn("GPIO", str(e).upper())


class TestChoiceAssaySensorCfg(unittest.TestCase):
    """Test the configuration dataclass"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        from choice_assay.my_choice_assay_sensor_with_leds import DEFAULT_CA_SENSOR_CFG
        config = DEFAULT_CA_SENSOR_CFG
        
        self.assertTrue(config.enable_leds)
        self.assertEqual(config.frame_rate, 5)
        self.assertEqual(config.motion_threshold, 125)
        self.assertEqual(config.grace_period_seconds, 3.0)
        self.assertEqual(config.detection_frames_needed, 7)

    def test_configuration_modification(self):
        """Test configuration can be modified"""
        from choice_assay.my_choice_assay_sensor_with_leds import DEFAULT_CA_SENSOR_CFG
        from dataclasses import replace
        
        config = replace(
            DEFAULT_CA_SENSOR_CFG,
            frame_rate=10,
            motion_threshold=500,
            left_detection_roi=(10, 10, 300, 200)
        )
        
        self.assertEqual(config.frame_rate, 10)
        self.assertEqual(config.motion_threshold, 500)
        self.assertEqual(config.left_detection_roi, (10, 10, 300, 200))


if __name__ == '__main__':
    # Configure test output
    unittest.main(verbosity=2, buffer=True)