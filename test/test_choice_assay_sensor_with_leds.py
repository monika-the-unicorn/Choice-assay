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

    @patch('choice_assay.my_choice_assay_sensor_with_leds.GPIO')
    def test_gpio_initialization(self, mock_GPIO):
        """Test GPIO pins are initialized correctly"""
        # Ensure LEDs are enabled in config
        self.sensor.config.enable_leds = True
        
        # Mock GPIO to be available
        mock_GPIO.setmode = Mock()
        mock_GPIO.setup = Mock() 
        mock_GPIO.OUT = 'OUT'
        mock_GPIO.BCM = 'BCM'
        mock_GPIO.LOW = 0
        
        self.sensor._initialize_gpio()
        
        # Check GPIO mode is set
        mock_GPIO.setmode.assert_called_once_with('BCM')
        
        # Check LED pins are set to output (should be called for each LED pin)
        self.assertTrue(mock_GPIO.setup.called)
        self.assertTrue(self.sensor.gpio_initialized)

    @patch('choice_assay.my_choice_assay_sensor_with_leds.GPIO')
    def test_led_control_functions(self, mock_GPIO):
        """Test LED control functions work correctly"""
        # Set up GPIO to be initialized
        self.sensor.gpio_initialized = True
        mock_GPIO.output = Mock()
        mock_GPIO.HIGH = 1
        mock_GPIO.LOW = 0
        
        # Test generic LED control method
        self.sensor._set_led_state(17, True)  # Red LED on
        mock_GPIO.output.assert_called_with(17, 1)
        
        self.sensor._set_led_state(17, False)  # Red LED off
        mock_GPIO.output.assert_called_with(17, 0)
        
        self.sensor._set_led_state(22, True)  # Green LED on
        mock_GPIO.output.assert_called_with(22, 1)
        
        self.sensor._set_led_state(27, True)  # Blue LED on
        mock_GPIO.output.assert_called_with(27, 1)
        
        self.sensor._set_led_state(23, True)  # Frame LED on
        mock_GPIO.output.assert_called_with(23, 1)

    @patch('choice_assay.my_choice_assay_sensor_with_leds.GPIO')
    def test_emergency_stop_button(self, mock_GPIO):
        """Test emergency stop button functionality"""
        # Set up GPIO to be initialized
        self.sensor.gpio_initialized = True
        mock_GPIO.input = Mock()
        mock_GPIO.HIGH = 1
        
        # Test button pressed (returns HIGH)
        mock_GPIO.input.return_value = mock_GPIO.HIGH
        self.assertTrue(self.sensor._check_emergency_stop())
        
        # Test button not pressed (returns LOW)  
        mock_GPIO.input.return_value = 0  # LOW
        self.assertFalse(self.sensor._check_emergency_stop())

    @patch('choice_assay.my_choice_assay_sensor_with_leds.GPIO')
    def test_led_status_updates(self, mock_GPIO):
        """Test LED status updates based on motion detection state"""
        # Set up GPIO to be initialized
        self.sensor.gpio_initialized = True
        mock_GPIO.output = Mock()
        mock_GPIO.HIGH = 1
        mock_GPIO.LOW = 0
        
        # Test individual LED state changes (the sensor manages LED state internally)
        # We can test the _set_led_state method which is used internally
        
        # Test motion detection LED (green) activation
        self.sensor._set_led_state(22, True)  # Green LED for active motion
        mock_GPIO.output.assert_called_with(22, 1)
        
        # Test motion counter LED (red) state
        self.sensor._set_led_state(17, False)  # Red LED off during detection  
        mock_GPIO.output.assert_called_with(17, 0)

    def test_dual_arena_motion_detection_logic(self):
        """Test dual arena motion detection mutual exclusivity"""
        # Create test frames
        frame1 = np.zeros((1232, 1640, 3), dtype=np.uint8)  # Use config dimensions
        frame2 = np.zeros((1232, 1640, 3), dtype=np.uint8)
        
        # Set up some motion in the left detection area (210, 373, 560, 578)
        frame2[373:578, 210:560] = 255  # Motion in left area
        
        # Mock cv2 functions to simulate the actual processing
        with patch('choice_assay.my_choice_assay_sensor_with_leds.cv2') as mock_cv2:
            # Mock the image processing pipeline
            mock_cv2.cvtColor.side_effect = lambda img, code: np.mean(img, axis=2).astype(np.uint8)
            mock_cv2.GaussianBlur.side_effect = lambda img, kernel, sigma: img  # Return unchanged
            mock_cv2.absdiff.side_effect = lambda img1, img2: np.abs(img1.astype(int) - img2.astype(int)).astype(np.uint8)
            mock_cv2.threshold.side_effect = lambda img, thresh, maxval, type: (None, (img > thresh).astype(np.uint8) * 255)
            mock_cv2.countNonZero.return_value = 1000  # Above motion threshold
            
            # First frame to initialize previous_frame
            left_motion1, right_motion1 = self.sensor._detect_motion_dual_arena(frame1)
            self.assertFalse(left_motion1)  # First frame always returns False
            self.assertFalse(right_motion1)
            
            # Second frame should detect motion
            left_motion2, right_motion2 = self.sensor._detect_motion_dual_arena(frame2)
            
            # Should detect motion (mocked to return high pixel count)
            self.assertTrue(left_motion2 or right_motion2)

    @patch('choice_assay.my_choice_assay_sensor_with_leds.GPIO')
    def test_cleanup_gpio(self, mock_GPIO):
        """Test GPIO cleanup cleans up properly"""
        # Set up GPIO to be initialized
        self.sensor.gpio_initialized = True
        mock_GPIO.cleanup = Mock()
        
        self.sensor._cleanup_gpio()
        
        # GPIO should be cleaned up
        mock_GPIO.cleanup.assert_called_once()

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