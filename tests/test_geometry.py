"""
test_geometry.py
-----------------
Tests for app/analytics/geometry.py - pure math functions with no
dependency on OpenCV, YOLO, or any model. This is exactly why these are
easy and fast to unit test: given known inputs, we know the correct
output ahead of time, unlike a model's prediction which is probabilistic.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.analytics.geometry import point_in_roi, has_crossed_line


class TestPointInROI:
    def setup_method(self):
        self.roi = {"x1": 100, "y1": 100, "x2": 300, "y2": 300}

    def test_point_clearly_inside(self):
        assert point_in_roi((200, 200), self.roi) is True

    def test_point_clearly_outside(self):
        assert point_in_roi((50, 50), self.roi) is False

    def test_point_on_boundary_counts_as_inside(self):
        assert point_in_roi((100, 100), self.roi) is True
        assert point_in_roi((300, 300), self.roi) is True

    def test_point_just_outside_boundary(self):
        assert point_in_roi((301, 200), self.roi) is False
        assert point_in_roi((200, 99), self.roi) is False


class TestLineCrossing:
    def setup_method(self):
        self.line_start = (0, 200)
        self.line_end = (640, 200)

    def test_no_previous_point_returns_none(self):
        result = has_crossed_line(None, (100, 250), self.line_start, self.line_end)
        assert result is None

    def test_crossing_bottom_to_top_is_entry(self):
        prev = (100, 250)
        curr = (100, 150)
        result = has_crossed_line(prev, curr, self.line_start, self.line_end)
        assert result == "entry"

    def test_crossing_top_to_bottom_is_exit(self):
        prev = (100, 150)
        curr = (100, 250)
        result = has_crossed_line(prev, curr, self.line_start, self.line_end)
        assert result == "exit"

    def test_no_crossing_when_staying_on_same_side(self):
        prev = (100, 150)
        curr = (150, 160)
        result = has_crossed_line(prev, curr, self.line_start, self.line_end)
        assert result is None

    def test_small_jitter_near_line_does_not_double_count(self):
        prev = (100, 199)
        curr = (100, 201)
        result = has_crossed_line(prev, curr, self.line_start, self.line_end)
        assert result is None

    def test_genuine_crossing_still_detected_past_hysteresis_band(self):
        prev = (100, 220)
        curr = (100, 180)
        result = has_crossed_line(prev, curr, self.line_start, self.line_end)
        assert result == "entry"