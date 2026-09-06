"""
event_engine.py
---------------
A small, modular rule-based event system. Each rule is just a method that
looks at current tracks + history and decides whether to emit an Event.

Why rule-based (not ML) for this part:
- These are deterministic, explainable business rules ("if a person is in
  this rectangle, that's a restricted-zone violation") - not something
  that benefits from a learned model. Rule-based logic is also instantly
  explainable/auditable, which matters a lot in real surveillance systems
  (you must be able to say WHY an alert fired).
- Adding a new rule later (e.g., object count threshold) just means adding
  one more method here - existing rules aren't touched. This is what
  "modular" means in practice, not just a buzzword.
"""

import time
import logging

from app.analytics.geometry import point_in_roi, has_crossed_line

logger = logging.getLogger("smart_surveillance.events")


class Event:
    def __init__(self, event_type, track_id, class_name, message, timestamp=None):
        self.event_type = event_type
        self.track_id = track_id
        self.class_name = class_name
        self.message = message
        self.timestamp = timestamp or time.time()

    def __repr__(self):
        return f"Event({self.event_type}, track_id={self.track_id}, {self.message})"


class EventEngine:
    def __init__(self, config: dict, track_history):
        events_cfg = config.get("events", {})
        self.roi = events_cfg.get("roi")                     # {"x1":.., "y1":.., "x2":.., "y2":..} or None
        self.line = events_cfg.get("line")                   # {"start":[x,y], "end":[x,y]} or None
        self.loitering_seconds = events_cfg.get("loitering_seconds", 10)
        self.person_count_threshold = events_cfg.get("person_count_threshold", 5)

        self.track_history = track_history

        # State that must persist across frames (to avoid duplicate counting)
        self.entry_count = 0
        self.exit_count = 0
        self._already_loitering_alerted = set()   # track_ids already alerted, avoid spam
        self._already_roi_alerted = set()

    def evaluate(self, tracked_objects, frame_number) -> list:
        """
        Runs all rules against the current frame's tracked objects.
        Returns a list of Event objects generated this frame.
        """
        events = []

        events.extend(self._check_line_crossing(tracked_objects))
        events.extend(self._check_roi_violation(tracked_objects))
        events.extend(self._check_loitering(tracked_objects))
        events.extend(self._check_count_threshold(tracked_objects))

        return events

    # ---------- Rule 1: Line crossing ----------
    def _check_line_crossing(self, tracked_objects) -> list:
        events = []
        if not self.line:
            return events

        line_start = tuple(self.line["start"])
        line_end = tuple(self.line["end"])

        for obj in tracked_objects:
            prev_center = self.track_history.get_previous_center(obj.track_id)
            curr_center = obj.center

            direction = has_crossed_line(prev_center, curr_center, line_start, line_end)
            if direction == "entry":
                self.entry_count += 1
                events.append(Event(
                    "line_crossing_entry", obj.track_id, obj.class_name,
                    f"{obj.class_name} (ID {obj.track_id}) entered via line crossing"
                ))
                logger.info(f"EVENT: line entry by track {obj.track_id}")
            elif direction == "exit":
                self.exit_count += 1
                events.append(Event(
                    "line_crossing_exit", obj.track_id, obj.class_name,
                    f"{obj.class_name} (ID {obj.track_id}) exited via line crossing"
                ))
                logger.info(f"EVENT: line exit by track {obj.track_id}")

        return events

    # ---------- Rule 2: Restricted ROI violation ----------
    def _check_roi_violation(self, tracked_objects) -> list:
        events = []
        if not self.roi:
            return events

        for obj in tracked_objects:
            if obj.class_name != "person":
                continue  # restricted-zone rule only cares about people

            if point_in_roi(obj.center, self.roi):
                if obj.track_id not in self._already_roi_alerted:
                    self._already_roi_alerted.add(obj.track_id)
                    events.append(Event(
                        "restricted_zone_entry", obj.track_id, obj.class_name,
                        f"Person (ID {obj.track_id}) entered restricted zone"
                    ))
                    logger.info(f"EVENT: restricted zone entry by track {obj.track_id}")
            else:
                # left the zone - allow re-alerting if they re-enter later
                self._already_roi_alerted.discard(obj.track_id)

        return events

    # ---------- Rule 3: Loitering ----------
    def _check_loitering(self, tracked_objects) -> list:
        events = []

        for obj in tracked_objects:
            first_seen = self.track_history.get_first_seen_time(obj.track_id)
            if first_seen is None:
                continue

            duration = time.time() - first_seen
            if duration >= self.loitering_seconds and obj.track_id not in self._already_loitering_alerted:
                self._already_loitering_alerted.add(obj.track_id)
                events.append(Event(
                    "loitering", obj.track_id, obj.class_name,
                    f"{obj.class_name} (ID {obj.track_id}) has been present for "
                    f"{duration:.0f}s (threshold: {self.loitering_seconds}s)"
                ))
                logger.info(f"EVENT: loitering by track {obj.track_id} ({duration:.0f}s)")

        return events

    # ---------- Rule 4: Object count threshold ----------
    def _check_count_threshold(self, tracked_objects) -> list:
        events = []
        person_count = sum(1 for obj in tracked_objects if obj.class_name == "person")

        if person_count > self.person_count_threshold:
            events.append(Event(
                "count_threshold_exceeded", None, "person",
                f"Person count ({person_count}) exceeded threshold "
                f"({self.person_count_threshold})"
            ))
            logger.warning(f"EVENT: person count threshold exceeded ({person_count})")

        return events
