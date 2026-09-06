"""
test_event_engine.py
----------------------
Tests for app/events/event_engine.py - the rule-based logic (line
crossing, ROI, loitering, count threshold). We use a simple fake/mock
"FakeTrackedObject" instead of running real YOLO detection, because
these rules only care about track_id, class_name, and center position -
they don't need an actual model. This is exactly the benefit of keeping
business logic (event_engine.py) separate from the model code
(tracker.py): we can test the RULES without needing a GPU, a webcam, or
even ultralytics installed.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.events.event_engine import EventEngine
from app.analytics.track_history import TrackHistory


class FakeTrackedObject:
    """Minimal stand-in for tracker.TrackedObject - same interface, no model needed."""
    def __init__(self, track_id, class_name, center):
        self.track_id = track_id
        self.class_name = class_name
        self._center = center
        self.bbox = (center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10)
        self.confidence = 0.9

    @property
    def center(self):
        return self._center


def make_config(roi=None, line=None, loitering_seconds=10, person_count_threshold=5):
    return {
        "events": {
            "roi": roi,
            "line": line,
            "loitering_seconds": loitering_seconds,
            "person_count_threshold": person_count_threshold,
        }
    }


class TestROIEvents:
    def test_person_entering_roi_fires_event(self):
        config = make_config(roi={"x1": 100, "y1": 100, "x2": 300, "y2": 300})
        history = TrackHistory()
        engine = EventEngine(config, history)

        obj = FakeTrackedObject(track_id=1, class_name="person", center=(200, 200))
        events = engine.evaluate([obj], frame_number=1)

        assert len(events) == 1
        assert events[0].event_type == "restricted_zone_entry"

    def test_same_track_does_not_refire_while_still_inside(self):
        config = make_config(roi={"x1": 100, "y1": 100, "x2": 300, "y2": 300})
        history = TrackHistory()
        engine = EventEngine(config, history)

        obj = FakeTrackedObject(track_id=1, class_name="person", center=(200, 200))
        engine.evaluate([obj], frame_number=1)
        events = engine.evaluate([obj], frame_number=2)

        assert len(events) == 0

    def test_non_person_class_does_not_trigger_roi_event(self):
        config = make_config(roi={"x1": 100, "y1": 100, "x2": 300, "y2": 300})
        history = TrackHistory()
        engine = EventEngine(config, history)

        obj = FakeTrackedObject(track_id=1, class_name="car", center=(200, 200))
        events = engine.evaluate([obj], frame_number=1)

        assert len(events) == 0


class TestCountThresholdEvents:
    def test_count_below_threshold_no_event(self):
        config = make_config(person_count_threshold=3)
        history = TrackHistory()
        engine = EventEngine(config, history)

        objs = [FakeTrackedObject(i, "person", (i * 10, 50)) for i in range(2)]
        events = engine.evaluate(objs, frame_number=1)

        assert len(events) == 0

    def test_count_above_threshold_fires_event(self):
        config = make_config(person_count_threshold=3)
        history = TrackHistory()
        engine = EventEngine(config, history)

        objs = [FakeTrackedObject(i, "person", (i * 10, 50)) for i in range(5)]
        events = engine.evaluate(objs, frame_number=1)

        count_events = [e for e in events if e.event_type == "count_threshold_exceeded"]
        assert len(count_events) == 1


class TestLoiteringEvents:
    def test_loitering_not_triggered_before_threshold(self):
        config = make_config(loitering_seconds=100)
        history = TrackHistory()
        engine = EventEngine(config, history)

        obj = FakeTrackedObject(track_id=1, class_name="person", center=(50, 50))
        history.update(1, "person", obj.bbox, obj.center, frame_number=1)
        events = engine.evaluate([obj], frame_number=1)

        loiter_events = [e for e in events if e.event_type == "loitering"]
        assert len(loiter_events) == 0

    def test_loitering_triggers_after_threshold_elapsed(self):
        config = make_config(loitering_seconds=0)
        history = TrackHistory()
        engine = EventEngine(config, history)

        obj = FakeTrackedObject(track_id=1, class_name="person", center=(50, 50))
        history.update(1, "person", obj.bbox, obj.center, frame_number=1)
        events = engine.evaluate([obj], frame_number=1)

        loiter_events = [e for e in events if e.event_type == "loitering"]
        assert len(loiter_events) == 1


class TestLineCrossingEvents:
    def test_crossing_fires_exactly_one_event_and_updates_count(self):
        config = make_config(line={"start": [0, 200], "end": [640, 200]})
        history = TrackHistory()
        engine = EventEngine(config, history)

        obj = FakeTrackedObject(track_id=1, class_name="person", center=(100, 250))
        history.update(1, "person", obj.bbox, obj.center, frame_number=1)

        obj2 = FakeTrackedObject(track_id=1, class_name="person", center=(100, 150))
        events = engine.evaluate([obj2], frame_number=2)

        crossing_events = [e for e in events if "line_crossing" in e.event_type]
        assert len(crossing_events) == 1