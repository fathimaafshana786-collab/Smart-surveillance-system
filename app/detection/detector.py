"""
detector.py
-----------
Wraps a YOLO model so the rest of the pipeline works with a simple,
consistent output format regardless of which detector we use underneath.

Why wrap the model instead of calling `model(frame)` directly in app.py:
- If we later swap YOLOv8 for a different model (or add ONNX Runtime /
  OpenVINO inference), only this file changes.
- We control exactly what gets returned: a clean list of detections with
  class name, confidence, and box coordinates - not whatever raw object
  the underlying library happens to produce.

Key concepts (kept short here - happy to go deeper if you want):
- Bounding box: rectangle (x1, y1, x2, y2) marking where an object is.
- Confidence score: model's own certainty that the box contains that class.
- Confidence threshold: we discard detections below this score as noise.
- NMS (Non-Max Suppression) + IoU threshold: when the model draws several
  overlapping boxes for the same object, NMS keeps only the best one and
  removes the rest based on how much they overlap (IoU = Intersection
  over Union of two boxes). YOLO does this internally - we just configure
  the threshold.
"""

import logging
from ultralytics import YOLO

logger = logging.getLogger("smart_surveillance.detection")


class Detection:
    """A single detected object in one frame."""
    def __init__(self, class_name, confidence, bbox):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2) in pixel coordinates

    def __repr__(self):
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


class ObjectDetector:
    def __init__(self, config: dict):
        model_cfg = config["model"]
        self.weights_path = model_cfg["weights_path"]
        self.conf_threshold = model_cfg["confidence_threshold"]
        self.iou_threshold = model_cfg["iou_threshold"]
        self.allowed_classes = set(model_cfg["classes"])

        logger.info(f"Loading YOLO model from: {self.weights_path}")
        try:
            self.model = YOLO(self.weights_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # COCO class id -> name mapping, provided by the model itself
        self.class_names = self.model.names
        logger.info("Model loaded successfully.")

    def detect(self, frame) -> list:
        """
        Runs inference on a single frame.
        Returns a list of Detection objects, already filtered by
        confidence threshold and by our allowed class list.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]

                if class_name not in self.allowed_classes:
                    continue  # not a class we care about for surveillance

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(Detection(class_name, confidence, (x1, y1, x2, y2)))

        return detections
