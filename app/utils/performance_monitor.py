"""
performance_monitor.py
-----------------------
Measures REAL timing of each pipeline stage. Nothing here is invented or
estimated - every number comes from actually timing the code with
Python's time.perf_counter(), which is the standard high-resolution timer
for measuring code performance (more precise than time.time() for short
durations).

Why measure multiple separate stages instead of just one overall FPS:
- "Model inference time" = how long YOLO's forward pass itself takes.
- "End-to-end pipeline latency" = capture + resize + inference + tracking
  + event logic + drawing - EVERYTHING for one frame, start to finish.
These are genuinely different numbers, and confusing them is a common
mistake. A model might report "10ms inference" in isolation, but if your
pipeline's total per-frame time is 40ms, your real FPS is bounded by 40ms,
not 10ms. Interviewers who know CV will often specifically probe whether
you understand this distinction.
"""

import time
from collections import deque


class PerformanceMonitor:
    def __init__(self, window_size: int = 30):
        # Rolling windows so displayed numbers reflect RECENT performance,
        # not a single lucky/unlucky frame or a stale average from minutes ago.
        self.window_size = window_size
        self._preprocess_times = deque(maxlen=window_size)
        self._inference_times = deque(maxlen=window_size)
        self._postprocess_times = deque(maxlen=window_size)
        self._total_times = deque(maxlen=window_size)

        self._stage_start = None
        self._frame_start = None

    def start_frame(self):
        self._frame_start = time.perf_counter()

    def mark_preprocess_start(self):
        self._stage_start = time.perf_counter()

    def mark_preprocess_end(self):
        self._preprocess_times.append(time.perf_counter() - self._stage_start)

    def mark_inference_start(self):
        self._stage_start = time.perf_counter()

    def mark_inference_end(self):
        self._inference_times.append(time.perf_counter() - self._stage_start)

    def mark_postprocess_start(self):
        self._stage_start = time.perf_counter()

    def mark_postprocess_end(self):
        self._postprocess_times.append(time.perf_counter() - self._stage_start)

    def end_frame(self):
        if self._frame_start is not None:
            self._total_times.append(time.perf_counter() - self._frame_start)

    @staticmethod
    def _avg_ms(values) -> float:
        if not values:
            return 0.0
        return (sum(values) / len(values)) * 1000.0

    def get_metrics(self) -> dict:
        """
        Returns average timings (in milliseconds) over the recent rolling
        window, plus derived FPS. All real, measured values - nothing
        hardcoded or assumed.
        """
        avg_preprocess_ms = self._avg_ms(self._preprocess_times)
        avg_inference_ms = self._avg_ms(self._inference_times)
        avg_postprocess_ms = self._avg_ms(self._postprocess_times)
        avg_total_ms = self._avg_ms(self._total_times)

        # FPS derived from actual measured end-to-end latency, not a
        # separately-tracked frame counter - this keeps the two numbers
        # mathematically consistent with each other.
        fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

        return {
            "preprocess_ms": round(avg_preprocess_ms, 2),
            "inference_ms": round(avg_inference_ms, 2),
            "postprocess_ms": round(avg_postprocess_ms, 2),
            "total_latency_ms": round(avg_total_ms, 2),
            "fps": round(fps, 1),
        }

    def get_resource_usage(self) -> dict:
        """
        CPU and memory usage of THIS process, using psutil (already a
        dependency of ultralytics, so no new install needed).
        GPU usage is intentionally not reported here unless a GPU is
        actually detected - we don't want to display a fake/zero GPU
        stat on a CPU-only laptop and imply GPU usage was measured.
        """
        try:
            import psutil
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=None)
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return {
                "cpu_percent": round(cpu_percent, 1),
                "memory_mb": round(memory_mb, 1),
            }
        except Exception:
            return {"cpu_percent": None, "memory_mb": None}