"""
track_history.py
-----------------
Maintains a short history of positions for every active track_id, across
frames. This is the "temporal information" piece of the pipeline - without
it, every frame is analyzed in isolation and we could never answer
questions like "did this object cross the line?" or "has it been in this
zone too long?" because those questions inherently need to compare NOW
against the PAST.

Design choice: we store history per track_id in a plain dict of lists,
capped at a max length (deque with maxlen) so memory doesn't grow forever
for a track that lives a long time.
"""

import time
import logging
from collections import deque

logger = logging.getLogger("smart_surveillance.analytics")


class TrackHistory:
    def __init__(self, max_history_per_track: int = 300):
        self._history = {}
        self.max_history_per_track = max_history_per_track

    def update(self, track_id: int, class_name: str, bbox, center, frame_number: int):
        entry = {
            "timestamp": time.time(),
            "frame_number": frame_number,
            "bbox": bbox,
            "center": center,
            "class_name": class_name,
        }

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.max_history_per_track)

        self._history[track_id].append(entry)

    def get_history(self, track_id: int):
        return list(self._history.get(track_id, []))

    def get_previous_center(self, track_id: int):
        """
        Returns the most recently recorded center for this track, or None
        if we have no history yet.

        Important design note: in the real pipeline (app.py/dashboard.py),
        event_engine.evaluate() is called BEFORE track_history.update()
        for the current frame. That means at the moment we check for a
        line crossing, history only contains PAST frames - the most
        recent stored entry (hist[-1]) IS the previous frame's position,
        and the current frame's position lives on the tracked object
        itself (not yet in history). So we only need ONE stored entry to
        return a valid "previous center," not two.
        """
        hist = self._history.get(track_id)
        if not hist:
            return None
        return hist[-1]["center"]

    def get_first_seen_time(self, track_id: int):
        """Used for loitering - how long has this track existed."""
        hist = self._history.get(track_id)
        if not hist:
            return None
        return hist[0]["timestamp"]

    def cleanup_stale_tracks(self, active_track_ids: set, max_age_seconds: float = 5.0):
        """
        Removes history for tracks that haven't appeared in a while (object
        left the frame permanently, not just briefly occluded). Prevents
        memory from growing unboundedly over a long-running video/webcam session.
        """
        now = time.time()
        stale_ids = []
        for track_id, hist in self._history.items():
            if track_id in active_track_ids:
                continue
            if hist and (now - hist[-1]["timestamp"]) > max_age_seconds:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._history[track_id]
            logger.debug(f"Removed stale track history for track_id={track_id}")