"""
dashboard.py
------------
Streamlit-based dashboard: live video with detections/tracks/events,
real performance metrics, and a downloadable event log.

Run with:  streamlit run dashboard.py

How Streamlit's execution model works (important to understand, not just
copy-paste): Streamlit re-runs this ENTIRE script top-to-bottom on every
interaction. For a continuously-updating video feed, we can't just print
frames one after another like in a normal script - instead we create
"placeholder" containers (st.empty()) once, then repeatedly overwrite
their contents inside a while loop. This is the standard pattern for
video/live-updating content in Streamlit.
"""

import time
import csv
import os
import cv2
import streamlit as st
import pandas as pd

from app.utils.config_loader import load_config, ConfigError
from app.utils.logger_setup import setup_logging
from app.utils.performance_monitor import PerformanceMonitor
from app.video.video_source import VideoSource, VideoSourceError
from app.tracking.tracker import ObjectTracker
from app.analytics.track_history import TrackHistory
from app.events.event_engine import EventEngine


st.set_page_config(page_title="Smart Surveillance Dashboard", layout="wide")


def draw_tracks(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        label = f"ID {obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
        cv2.line(frame, tuple(line["start"]), tuple(line["end"]), (0, 255, 255), 2)
    return frame


@st.cache_resource
def init_pipeline():
    """
    Cached so the model only loads ONCE across Streamlit's re-runs, not
    every time the script re-executes (which would be very slow -
    reloading a YOLO model on every interaction is wasteful).
    """
    config = load_config("config/config.yaml")
    logger = setup_logging(
        log_dir=config["output"]["log_dir"],
        level=config["logging"]["level"]
    )
    tracker = ObjectTracker(config)
    track_history = TrackHistory()
    event_engine = EventEngine(config, track_history)
    perf_monitor = PerformanceMonitor()
    return config, logger, tracker, track_history, event_engine, perf_monitor


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
    st.title("🎥 Smart Surveillance Video Analytics")

    try:
        config, logger, tracker, track_history, event_engine, perf_monitor = init_pipeline()
    except ConfigError as e:
        st.error(f"Config error: {e}")
        return
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return

    # Sidebar controls
    st.sidebar.header("Controls")
    run_button = st.sidebar.checkbox("Start / Resume processing", value=False)
    performance_mode = st.sidebar.checkbox(
        "Performance mode (hide overlays for higher FPS)", value=False
    )

    # Layout: video on the left, metrics on the right
    col_video, col_metrics = st.columns([2, 1])

    with col_video:
        video_placeholder = st.empty()

    with col_metrics:
        st.subheader("Live Metrics")
        metrics_placeholder = st.empty()
        st.subheader("Event Counts")
        counts_placeholder = st.empty()

    st.subheader("Event Log")
    event_log_placeholder = st.empty()
    download_placeholder = st.empty()

    if "session_events" not in st.session_state:
        st.session_state.session_events = []

    if not run_button:
        st.info("Check 'Start / Resume processing' in the sidebar to begin.")
        return

    try:
        video_source = VideoSource(config)
    except VideoSourceError as e:
        st.error(f"Video source error: {e}")
        return

    csv_file, csv_writer = init_event_log(config["output"]["log_dir"])
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]
    frame_count = 0

    try:
        while run_button:
            perf_monitor.start_frame()

            success, frame = video_source.read_frame()
            if not success:
                st.warning("Video ended or source disconnected.")
                break

            perf_monitor.mark_preprocess_start()
            frame = cv2.resize(frame, (resize_w, resize_h))
            frame_count += 1
            perf_monitor.mark_preprocess_end()

            perf_monitor.mark_inference_start()
            tracked_objects = tracker.track(frame)
            perf_monitor.mark_inference_end()

            perf_monitor.mark_postprocess_start()
            events = event_engine.evaluate(tracked_objects, frame_count)
            for obj in tracked_objects:
                track_history.update(obj.track_id, obj.class_name, obj.bbox, obj.center, frame_count)
            active_ids = {obj.track_id for obj in tracked_objects}
            track_history.cleanup_stale_tracks(active_ids)

            for event in events:
                csv_writer.writerow([event.timestamp, event.event_type, event.track_id,
                                      event.class_name, event.message])
                csv_file.flush()
                st.session_state.session_events.append({
                    "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
                    "event": event.event_type,
                    "object": event.class_name,
                    "track_id": event.track_id,
                })

            if not performance_mode:
                frame = draw_roi(frame, event_engine.roi)
                frame = draw_line(frame, event_engine.line)
                frame = draw_tracks(frame, tracked_objects)
            perf_monitor.mark_postprocess_end()

            perf_monitor.end_frame()

            # ---- Render everything into the placeholders ----
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            metrics = perf_monitor.get_metrics()
            resources = perf_monitor.get_resource_usage()
            metrics_placeholder.markdown(f"""
| Metric | Value |
|---|---|
| FPS | {metrics['fps']} |
| Total latency | {metrics['total_latency_ms']} ms |
| Inference time | {metrics['inference_ms']} ms |
| Preprocess time | {metrics['preprocess_ms']} ms |
| Postprocess time | {metrics['postprocess_ms']} ms |
| CPU usage | {resources['cpu_percent']}% |
| Memory usage | {resources['memory_mb']} MB |
""")

            counts_placeholder.markdown(f"""
| | |
|---|---|
| Entries | {event_engine.entry_count} |
| Exits | {event_engine.exit_count} |
| Active tracks | {len(tracked_objects)} |
""")

            if st.session_state.session_events:
                df = pd.DataFrame(st.session_state.session_events)
                event_log_placeholder.dataframe(df, use_container_width=True, height=250)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                download_placeholder.download_button(
                    "Download event log as CSV", csv_bytes,
                    file_name="event_log.csv", mime="text/csv",
                    key=f"download_btn_{frame_count}"
                )

            time.sleep(0.01)  # tiny yield so Streamlit's UI stays responsive

    except Exception as e:
        logger.error(f"Unexpected error during dashboard pipeline execution: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
    finally:
        csv_file.close()
        video_source.release()


if __name__ == "__main__":
    main()