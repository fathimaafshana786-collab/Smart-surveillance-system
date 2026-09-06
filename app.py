"""
app.py
------
Entry point for Chunk 2: adds tracking, ROI, line-crossing, and rule-based
events on top of Chunk 1's detection pipeline.

Pipeline now:
    Video Source -> Frame -> Tracker (detect+track) -> Track History update
    -> Event Engine (ROI / line / loitering / count) -> Draw everything
    -> Show on screen + log events

Run with:  python app.py
Press 'q' in the video window to quit.
"""

import time
import csv
import os
import cv2
import logging

from app.utils.config_loader import load_config, ConfigError
from app.utils.logger_setup import setup_logging
from app.video.video_source import VideoSource, VideoSourceError
from app.tracking.tracker import ObjectTracker
from app.analytics.track_history import TrackHistory
from app.events.event_engine import EventEngine


def draw_tracks(frame, tracked_objects):
    """Draws bounding boxes, track IDs, class names, and confidence."""
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        label = f"ID {obj.track_id} {obj.class_name} {obj.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        cv2.circle(frame, obj.center, 3, (0, 0, 255), -1)
    return frame


def draw_roi(frame, roi):
    if roi:
        cv2.rectangle(frame, (roi["x1"], roi["y1"]), (roi["x2"], roi["y2"]), (255, 0, 0), 2)
        cv2.putText(frame, "RESTRICTED ZONE", (roi["x1"], max(roi["y1"] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame


def draw_line(frame, line):
    if line:
        start = tuple(line["start"])
        end = tuple(line["end"])
        cv2.line(frame, start, end, (0, 255, 255), 2)
    return frame


def draw_stats(frame, entry_count, exit_count, active_tracks, fps):
    lines = [
        f"Entries: {entry_count}  Exits: {exit_count}",
        f"Active tracks: {active_tracks}",
        f"FPS: {fps:.1f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def init_event_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "event_log.csv")
    file_exists = os.path.exists(csv_path)

    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["timestamp", "event_type", "track_id", "class_name", "message"])
    return csv_file, writer


def main():
    try:
        config = load_config("config/config.yaml")
    except ConfigError as e:
        print(f"[FATAL] Config error: {e}")
        return

    logger = setup_logging(
        log_dir=config["output"]["log_dir"],
        level=config["logging"]["level"]
    )
    logger.info("Starting Smart Surveillance pipeline (Chunk 2: tracking + events)")

    try:
        video_source = VideoSource(config)
    except VideoSourceError as e:
        logger.error(f"Video source error: {e}")
        return

    try:
        tracker = ObjectTracker(config)
    except Exception as e:
        logger.error(f"Tracker failed to initialize: {e}")
        video_source.release()
        return

    track_history = TrackHistory()
    event_engine = EventEngine(config, track_history)
    csv_file, csv_writer = init_event_log(config["output"]["log_dir"])

    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]

    frame_count = 0
    fps_frame_counter = 0
    prev_time = time.time()
    current_fps = 0.0

    try:
        while True:
            success, frame = video_source.read_frame()
            if not success:
                logger.info("No more frames available. Exiting.")
                break

            frame = cv2.resize(frame, (resize_w, resize_h))
            frame_count += 1

            tracked_objects = tracker.track(frame)

            events = event_engine.evaluate(tracked_objects, frame_count)

            for obj in tracked_objects:
                track_history.update(
                    obj.track_id, obj.class_name, obj.bbox, obj.center, frame_count
                )

            active_ids = {obj.track_id for obj in tracked_objects}
            track_history.cleanup_stale_tracks(active_ids)

            for event in events:
                csv_writer.writerow([
                    event.timestamp, event.event_type, event.track_id,
                    event.class_name, event.message
                ])
                csv_file.flush()

            frame = draw_roi(frame, event_engine.roi)
            frame = draw_line(frame, event_engine.line)
            frame = draw_tracks(frame, tracked_objects)

            fps_frame_counter += 1
            current_time = time.time()
            elapsed = current_time - prev_time
            if elapsed >= 1.0:
                current_fps = fps_frame_counter / elapsed
                fps_frame_counter = 0
                prev_time = current_time

            frame = draw_stats(
                frame, event_engine.entry_count, event_engine.exit_count,
                len(tracked_objects), current_fps
            )

            cv2.imshow("Smart Surveillance", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit.")
                break
    except Exception as e:
        logger.error(f"Unexpected error during pipeline execution: {e}", exc_info=True)
    finally:
        csv_file.close()
        video_source.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline shutdown complete.")


if __name__ == "__main__":
    main()