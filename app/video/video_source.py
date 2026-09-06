"""
video_source.py
----------------
Wraps OpenCV's VideoCapture so the rest of the pipeline doesn't care
whether frames are coming from a webcam, a video file, or an RTSP stream.

Why wrap it instead of calling cv2.VideoCapture directly in app.py:
- app.py should only know "give me the next frame" - not HOW that frame
  was obtained. This is the "don't tightly couple input to the rest of
  the pipeline" principle.
- Centralizes error handling: if the webcam is busy or the file path is
  wrong, we want ONE clear error message, not a silent black frame.
"""

import logging
import cv2

logger = logging.getLogger("smart_surveillance.video")


class VideoSourceError(Exception):
    """Raised when the video source cannot be opened or read."""
    pass


class VideoSource:
    def __init__(self, config: dict):
        self.config = config["video"]
        self.source_type = self.config.get("source", "webcam")
        self.cap = None
        self._open()

    def _open(self):
        if self.source_type == "webcam":
            index = self.config.get("webcam_index", 0)
            logger.info(f"Opening webcam at index {index}")
            self.cap = cv2.VideoCapture(index)

        elif self.source_type == "file":
            path = self.config.get("file_path")
            logger.info(f"Opening video file: {path}")
            self.cap = cv2.VideoCapture(path)

        elif self.source_type == "rtsp":
            url = self.config.get("rtsp_url")
            logger.info(f"Opening RTSP stream: {url}")
            self.cap = cv2.VideoCapture(url)

        else:
            raise VideoSourceError(
                f"Unknown video source type: '{self.source_type}'. "
                f"Expected 'webcam', 'file', or 'rtsp'."
            )

        if self.cap is None or not self.cap.isOpened():
            raise VideoSourceError(
                f"Failed to open video source '{self.source_type}'. "
                f"Check that the webcam is connected, the file path is "
                f"correct, or the RTSP URL is reachable."
            )

    def read_frame(self):
        """
        Returns (success, frame). success=False means either the source
        failed or the video has ended - caller decides what to do.
        """
        if self.cap is None:
            raise VideoSourceError("Video source was never opened.")

        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to read frame - source may have ended or disconnected.")
        return success, frame

    def get_fps(self) -> float:
        """Native FPS reported by the source (not the same as our processing FPS)."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            logger.info("Video source released.")
