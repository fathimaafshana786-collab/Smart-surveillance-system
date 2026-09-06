"""
tracker.py
----------
Wraps YOLO + ByteTrack (ultralytics' built-in tracking mode) so the rest
of the pipeline gets back a clean list of TrackedObject, each carrying a
persistent track_id across frames - not just a one-off detection.

Why this is a separate module from detector.py, even though ultralytics
lets you call tracking directly on the model:
- Keeps the "detection only" pipeline (Chunk 1) and "tracking" pipeline
  (this chunk) cleanly separable - useful if we ever want to run
  detection-only for speed comparisons later (Phase 9 performance work).
- Gives us one place to attach OUR OWN position history per track_id,
  which detection/tracking libraries don't provide out of the box, and
  which we need for line-crossing, loitering, and trajectories later.
"""

import logging
from ultralytics import YOLO

logger = logging.getLogger("smart_surveillance.tracking")


class TrackedObject:
    """A single tracked object in the current frame, with a persistent ID."""
    def __init__(self, track_id, class_name, confidence, bbox):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def __repr__(self):
        return f"Track(id={self.track_id}, {self.class_name}, conf={self.confidence:.2f})"


class ObjectTracker:
    def __init__(self, config: dict):
        model_cfg = config["model"]
        self.weights_path = model_cfg["weights_path"]
        self.conf_threshold = model_cfg["confidence_threshold"]
        self.iou_threshold = model_cfg["iou_threshold"]
        self.allowed_classes = set(model_cfg["classes"])

        logger.info(f"Loading YOLO model for tracking from: {self.weights_path}")
        try:
            self.model = YOLO(self.weights_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.class_names = self.model.names
        logger.info("Tracking model loaded successfully (ByteTrack via ultralytics).")

    def track(self, frame) -> list:
        """
        Runs detection + tracking on a single frame.
        Returns a list of TrackedObject, each with a persistent track_id.

        Note: model.track() maintains internal tracker state ACROSS calls,
        so this must be called on consecutive frames of the same video,
        not on random/unrelated images.
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracked_objects = []
        for result in results:
            boxes = result.boxes
            if boxes.id is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                class_name = self.class_names[class_id]

                if class_name not in self.allowed_classes:
                    continue

                track_id = int(boxes.id[i])
                confidence = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                tracked_objects.append(
                    TrackedObject(track_id, class_name, confidence, (x1, y1, x2, y2))
                )

        return tracked_objects