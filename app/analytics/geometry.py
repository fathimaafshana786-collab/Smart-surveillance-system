"""
geometry.py
-----------
Pure geometry logic for:
1. Checking whether a point is inside a Region of Interest (ROI) rectangle.
2. Detecting whether a moving point has crossed a virtual line.

Kept separate from event_engine.py and completely independent of OpenCV/
YOLO so it's trivial to unit test (Phase: testing) - these are just math
functions on coordinates.

--- Line-crossing logic explained ---
A line is defined by two points: (x1, y1) and (x2, y2).
For any other point P, we can determine which SIDE of the line it's on
using the sign of the 2D cross product:

    cross = (x2 - x1) * (Py - y1) - (y2 - y1) * (Px - x1)

- cross > 0  -> point is on one side
- cross < 0  -> point is on the other side
- cross == 0 -> point is exactly ON the line

An object has CROSSED the line if its side flips between the previous
frame and the current frame (previous side was positive, now negative,
or vice versa). This is why we need the track's PREVIOUS center, not
just its current one - another reason temporal history matters.
"""

import math


def point_in_roi(point, roi: dict) -> bool:
    """
    roi is expected as: {"x1": int, "y1": int, "x2": int, "y2": int}
    representing a rectangle. Returns True if point is inside it.
    """
    x, y = point
    return roi["x1"] <= x <= roi["x2"] and roi["y1"] <= y <= roi["y2"]


def _line_side(point, line_start, line_end) -> float:
    """Returns signed value indicating which side of the line the point is on."""
    x1, y1 = line_start
    x2, y2 = line_end
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def has_crossed_line(prev_point, curr_point, line_start, line_end, min_distance: float = 4.0) -> str:
    """
    Returns:
        "entry"    if the point crossed from one side to the other in one direction
        "exit"     if it crossed in the opposite direction
        None       if no crossing occurred, or prev_point is unavailable

    Direction convention: side > 0 -> "outside", side < 0 -> "inside".
    Crossing from outside(+) to inside(-) = "entry".
    Crossing from inside(-) to outside(+) = "exit".

    min_distance is a HYSTERESIS BAND (in pixels). Why it's needed:
    a bounding box's center jitters slightly frame-to-frame even for a
    stationary or slow-moving object (the model's box isn't pixel-perfect
    identical every frame). If an object hovers right next to the line,
    that tiny jitter alone can flip its "side" back and forth, causing
    the same real crossing to be counted 2-3 times in a fraction of a
    second. Requiring the point to be at least `min_distance` pixels past
    the line (not just barely over it) filters out that jitter while still
    catching genuine crossings, which move much further than a few pixels
    between frames.
    """
    if prev_point is None:
        return None

    prev_side = _line_side(prev_point, line_start, line_end)
    curr_side = _line_side(curr_point, line_start, line_end)

    # Normalize by line length so min_distance is in real pixels, not
    # raw cross-product units (which scale with line length).
    line_length = math.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])
    if line_length == 0:
        return None

    prev_dist = prev_side / line_length
    curr_dist = curr_side / line_length

    # Require the point to be clearly on each side (past the hysteresis
    # band), not just barely over the line - this is what prevents
    # jitter-driven double counting.
    if abs(prev_dist) < min_distance or abs(curr_dist) < min_distance:
        return None

    if (prev_dist > 0 and curr_dist > 0) or (prev_dist < 0 and curr_dist < 0):
        return None

    if prev_dist > 0 and curr_dist < 0:
        return "entry"
    if prev_dist < 0 and curr_dist > 0:
        return "exit"

    return None