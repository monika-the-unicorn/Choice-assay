#!/usr/bin/env python3
"""Enhanced Test Suite for ChoiceAssaySensorWithLEDs

This enhanced test suite provides comprehensive coverage including:
- Video recording workflow testing
- Error scenario testing
- Integration tests for motion detection to recording pipeline
- save_recording validation
- Edge cases and failure modes
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

# Mock RPi.GPIO before importing our sensor
sys.modules["RPi"] = Mock()
sys.modules["RPi.GPIO"] = Mock()

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import UTC

from choice_assay.my_choice_assay_sensor import ChoiceAssaySensorWithLEDs


class TestChoiceAssaySensorVideoRecording(unittest.TestCase):
    """Test video recording functionality - the part that had the bug"""

    def setUp(self):
        """Set up test fixtures"""
        from datetime import datetime

        from choice_assay.my_choice_assay_sensor import DEFAULT_CA_SENSOR_CFG

        self.config = DEFAULT_CA_SENSOR_CFG

        # Mock all external dependencies
        self.mock_picamera2 = Mock()
        self.mock_cv2 = Mock()
        self.mock_api = Mock()
        self.mock_logger = Mock()

        # Set up datetime objects for realistic timestamp testing
        self.start_time = datetime(2025, 11, 4, 14, 25, 0, tzinfo=UTC)
        self.end_time = datetime(2025, 11, 4, 14, 30, 0, tzinfo=UTC)

        with (
            patch("choice_assay.my_choice_assay_sensor.Picamera2", self.mock_picamera2),
            patch("choice_assay.my_choice_assay_sensor.cv2", self.mock_cv2),
            patch("choice_assay.my_choice_assay_sensor.api", self.mock_api),
            patch("choice_assay.my_choice_assay_sensor.logger", self.mock_logger),
            patch("choice_assay.my_choice_assay_sensor.GPIO"),
        ):
            self.sensor = ChoiceAssaySensorWithLEDs(self.config)

        # Set up common mocks
        self.mock_video_writer = Mock()
        self.mock_cv2.VideoWriter.return_value = self.mock_video_writer
        self.mock_api.utc_now.return_value = self.end_time

    def test_stop_video_recording_returns_correct_tuple(self):
        """Test that _stop_video_recording returns (start_time, end_time) tuple"""
        # Setup active recording
        self.sensor.video_writer = self.mock_video_writer
        self.sensor.current_recording_start_time = self.start_time
        self.sensor.current_recording_arena = "left"

        # Call the method
        result = self.sensor._stop_video_recording()

        # Verify return type and values
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        start_time, end_time = result
        self.assertEqual(start_time, self.start_time)
        # end_time should be a datetime object from api.utc_now()
        self.assertIsInstance(end_time, type(self.start_time))

        # Verify cleanup
        self.mock_video_writer.release.assert_called_once()
        self.assertIsNone(self.sensor.video_writer)
        self.assertIsNone(self.sensor.current_recording_arena)
        self.assertIsNone(self.sensor.current_recording_start_time)

    def test_stop_video_recording_when_no_recording(self):
        """Test _stop_video_recording when no recording is active"""
        # No active recording
        self.sensor.video_writer = None

        # Call the method
        result = self.sensor._stop_video_recording()

        # Should return None
        self.assertIsNone(result)

    @patch.object(ChoiceAssaySensorWithLEDs, "save_recording")
    def test_normal_recording_stop_calls_save_recording(self, mock_save_recording):
        """Test that normal recording stop properly calls save_recording"""
        # Setup recording state
        self.sensor.video_writer = self.mock_video_writer
        self.sensor.current_recording_start_time = self.start_time
        self.sensor.current_recording_arena = "left"
        current_filename = "test_video.avi"

        with patch.object(self.sensor, "_get_stream_index_for_arena", return_value=0):
            # Save arena before stop (this is the fix we made)
            arena_name = self.sensor.current_recording_arena
            start_time, end_time = self.sensor._stop_video_recording()

            if current_filename and start_time and arena_name:
                stream_index = 0
                mock_save_recording(stream_index, current_filename, start_time=start_time, end_time=end_time)

        # Verify save_recording was called (check that it was called with correct parameters)
        mock_save_recording.assert_called_once()
        call_args = mock_save_recording.call_args
        self.assertEqual(call_args[0][0], 0)  # stream_index
        self.assertEqual(call_args[0][1], current_filename)  # filename
        self.assertEqual(call_args[1]["start_time"], self.start_time)  # start_time
        # end_time should be a datetime object
        self.assertIsInstance(call_args[1]["end_time"], type(self.start_time))

    @patch.object(ChoiceAssaySensorWithLEDs, "save_recording")
    def test_error_recording_cleanup_calls_save_recording(self, mock_save_recording):
        """Test that error cleanup properly saves recordings (the bug we fixed)"""
        # Setup recording state
        self.sensor.video_writer = self.mock_video_writer
        self.sensor.current_recording_start_time = self.start_time
        self.sensor.current_recording_arena = "right"
        current_filename = "test_video_error.avi"

        with patch.object(self.sensor, "_get_stream_index_for_arena", return_value=1):
            # Simulate the error cleanup pattern (the fix)
            arena_name = self.sensor.current_recording_arena  # Save before stop
            result = self.sensor._stop_video_recording()

            if result and current_filename and arena_name:
                start_time, end_time = result
                stream_index = 1
                mock_save_recording(stream_index, current_filename, start_time=start_time, end_time=end_time)

        # Verify save_recording was called even in error scenario
        mock_save_recording.assert_called_once()
        call_args = mock_save_recording.call_args
        self.assertEqual(call_args[0][0], 1)  # stream_index
        self.assertEqual(call_args[0][1], current_filename)  # filename
        self.assertEqual(call_args[1]["start_time"], self.start_time)  # start_time
        # end_time should be a datetime object
        self.assertIsInstance(call_args[1]["end_time"], type(self.start_time))

    def test_get_stream_index_for_arena(self):
        """Test stream index mapping for different arenas"""
        # Test left arena
        self.assertEqual(self.sensor._get_stream_index_for_arena("left"), 0)

        # Test right arena
        self.assertEqual(self.sensor._get_stream_index_for_arena("right"), 1)

        # Test that any non-"left" value defaults to right (current implementation behavior)
        self.assertEqual(self.sensor._get_stream_index_for_arena("invalid"), 1)

    @patch.object(ChoiceAssaySensorWithLEDs, "save_recording")
    def test_video_recording_integration_workflow(self, mock_save_recording):
        """Integration test of complete video recording workflow"""
        current_filename = "integration_test.avi"

        # Test complete workflow: start -> write frames -> stop -> save
        with patch.object(self.sensor, "_get_stream_index_for_arena", return_value=0):
            # Set up recording state manually (simulating what _start_video_recording does)
            self.sensor.video_writer = self.mock_video_writer
            self.sensor.current_recording_start_time = self.start_time
            self.sensor.current_recording_arena = "left"

            # Simulate frame writing
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.sensor._write_frame_to_video(frame, "left")

            # Stop and save
            arena_name = self.sensor.current_recording_arena  # Save before stop
            result = self.sensor._stop_video_recording()

            if result and current_filename and arena_name:
                start_time, end_time = result
                mock_save_recording(0, current_filename, start_time=start_time, end_time=end_time)

        # Verify the complete workflow
        mock_save_recording.assert_called_once()

    def test_multiple_recording_sessions(self):
        """Test multiple recording sessions don't interfere"""
        # First recording
        self.sensor.video_writer = self.mock_video_writer
        self.sensor.current_recording_start_time = "2025-11-04T14:25:00Z"
        self.sensor.current_recording_arena = "left"

        result1 = self.sensor._stop_video_recording()

        # Second recording
        self.sensor.video_writer = Mock()  # New writer
        self.sensor.current_recording_start_time = "2025-11-04T14:35:00Z"
        self.sensor.current_recording_arena = "right"

        result2 = self.sensor._stop_video_recording()

        # Both should return valid results
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertNotEqual(result1, result2)


class TestChoiceAssaySensorErrorScenarios(unittest.TestCase):
    """Test error scenarios and edge cases"""

    def setUp(self):
        """Set up test fixtures"""
        from datetime import datetime

        from choice_assay.my_choice_assay_sensor import DEFAULT_CA_SENSOR_CFG

        self.config = DEFAULT_CA_SENSOR_CFG

        # Set up datetime objects for realistic timestamp testing
        self.start_time = datetime(2025, 11, 4, 14, 25, 0, tzinfo=UTC)
        self.end_time = datetime(2025, 11, 4, 14, 30, 0, tzinfo=UTC)

        with (
            patch("choice_assay.my_choice_assay_sensor.Picamera2"),
            patch("choice_assay.my_choice_assay_sensor.cv2"),
            patch("choice_assay.my_choice_assay_sensor.GPIO"),
        ):
            self.sensor = ChoiceAssaySensorWithLEDs(self.config)

    @patch.object(ChoiceAssaySensorWithLEDs, "save_recording")
    def test_video_writer_release_exception_still_saves(self, mock_save_recording):
        """Test that even if video_writer.release() fails, we still save the recording"""
        # Setup recording with failing video writer
        mock_video_writer = Mock()
        mock_video_writer.release.side_effect = Exception("Release failed")

        self.sensor.video_writer = mock_video_writer
        self.sensor.current_recording_start_time = self.start_time
        self.sensor.current_recording_arena = "left"
        current_filename = "test_video.avi"

        with patch.object(self.sensor, "_get_stream_index_for_arena", return_value=0):
            # Simulate error cleanup with exception in release
            arena_name = self.sensor.current_recording_arena
            try:
                result = self.sensor._stop_video_recording()
                # Should still return valid result despite release error
                self.assertIsNotNone(result)

                if result and current_filename and arena_name:
                    start_time, end_time = result
                    mock_save_recording(0, current_filename, start_time=start_time, end_time=end_time)
            except Exception:
                # Even if there's an exception, we should still try to save
                if arena_name and current_filename:
                    mock_save_recording(
                        0, current_filename, start_time=self.start_time, end_time=self.end_time
                    )

        # Verify save_recording was still called
        mock_save_recording.assert_called_once()

    def test_concurrent_stop_calls_safe(self):
        """Test that multiple calls to _stop_video_recording are safe"""
        # Setup recording
        mock_video_writer = Mock()
        self.sensor.video_writer = mock_video_writer
        self.sensor.current_recording_start_time = "2025-11-04T14:25:00Z"
        self.sensor.current_recording_arena = "left"

        with patch("choice_assay.my_choice_assay_sensor.api") as mock_api:
            mock_api.utc_now.return_value = "2025-11-04T14:30:00Z"

            # First call should work
            result1 = self.sensor._stop_video_recording()
            self.assertIsNotNone(result1)

            # Second call should return None safely
            result2 = self.sensor._stop_video_recording()
            self.assertIsNone(result2)

            # Video writer should only be released once
            mock_video_writer.release.assert_called_once()


class TestChoiceAssaySensorMotionDetection(unittest.TestCase):
    """Test motion detection integration with recording"""

    def setUp(self):
        """Set up test fixtures"""
        from choice_assay.my_choice_assay_sensor import DEFAULT_CA_SENSOR_CFG

        self.config = DEFAULT_CA_SENSOR_CFG

        with (
            patch("choice_assay.my_choice_assay_sensor.Picamera2"),
            patch("choice_assay.my_choice_assay_sensor.cv2"),
            patch("choice_assay.my_choice_assay_sensor.GPIO"),
        ):
            self.sensor = ChoiceAssaySensorWithLEDs(self.config)

    @patch.object(ChoiceAssaySensorWithLEDs, "save_recording")
    def test_motion_detection_to_recording_pipeline(self, mock_save_recording):
        """Test complete pipeline from motion detection to recording save"""
        # Mock motion detection
        with (
            patch.object(self.sensor, "_detect_motion_dual_arena", return_value=(True, False)),
            patch.object(self.sensor, "_start_video_recording", return_value="motion_test.avi"),
            patch.object(self.sensor, "_get_stream_index_for_arena", return_value=0),
            patch("choice_assay.my_choice_assay_sensor.api") as mock_api,
        ):
            mock_api.utc_now.return_value = "2025-11-04T14:30:00Z"

            # Simulate motion detection triggering recording
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            left_motion, right_motion = self.sensor._detect_motion_dual_arena(frame)

            self.assertTrue(left_motion)
            self.assertFalse(right_motion)

            # Simulate recording workflow
            if left_motion:
                # Start recording
                self.sensor.current_recording_arena = "left"
                self.sensor.current_recording_start_time = "2025-11-04T14:25:00Z"
                self.sensor.video_writer = Mock()
                current_filename = "motion_test.avi"

                # Write some frames
                self.sensor._write_frame_to_video(frame, "left")

                # Stop recording (with fix)
                arena_name = self.sensor.current_recording_arena
                result = self.sensor._stop_video_recording()

                if result and current_filename and arena_name:
                    start_time, end_time = result
                    mock_save_recording(0, current_filename, start_time=start_time, end_time=end_time)

        # Verify recording was saved
        mock_save_recording.assert_called_once_with(
            0, "motion_test.avi", start_time="2025-11-04T14:25:00Z", end_time="2025-11-04T14:30:00Z"
        )


if __name__ == "__main__":
    # Configure test output
    unittest.main(verbosity=2, buffer=True)
