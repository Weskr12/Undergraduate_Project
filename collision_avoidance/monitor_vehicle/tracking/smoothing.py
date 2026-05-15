# -*- coding: utf-8 -*-
"""Per-track distance smoothing, approach velocity, and display EMA."""

from collections import deque

import numpy as np

from ..config import (
    APPROACH_LOOKBACK_FRAMES,
    BBOX_EMA_ALPHA,
    BEARING_EMA_ALPHA,
    DEPTH_DISPLAY_MAX_M,
    DISPLAY_DISTANCE_APPROACH_EMA_ALPHA,
    DISPLAY_DISTANCE_EMA_ALPHA,
    DISPLAY_DISTANCE_HISTORY_LEN,
    DISPLAY_DISTANCE_HOLD_FRAMES,
    DISPLAY_DISTANCE_MAX_APPROACH_STEP_M,
    DISPLAY_DISTANCE_MAX_RECEDING_STEP_M,
    DISPLAY_DISTANCE_MEDIAN_WINDOW,
    DISPLAY_DISTANCE_SMOOTHING_ENABLED,
    MIN_DEPTH,
    SMOOTH_WINDOW,
)
from ..detection.bbox import _bbox_xyxy_to_center_size, _center_size_to_bbox_xyxy
from ..utils import _ema, to_float_or_none


def _smoothed_distance_from_history(history):
    if not history:
        return None

    history_items = list(history)
    smooth_values = [value for _, value in history_items[-SMOOTH_WINDOW:]]
    if not smooth_values:
        return None
    return float(np.median(smooth_values))


def _finite_distance(value):
    value = to_float_or_none(value)
    if value is None:
        return None
    if not np.isfinite(value) or value <= MIN_DEPTH or value > DEPTH_DISPLAY_MAX_M:
        return None
    return float(value)


def _finite_fallback_distance(value):
    value = to_float_or_none(value)
    if value is None:
        return None
    if not np.isfinite(value) or value <= MIN_DEPTH:
        return None
    return float(value)


def _display_distance_source(detection_state):
    distance_source = str(detection_state.get("distance_source", ""))
    if distance_source == "yolo_size_fallback":
        return _finite_fallback_distance(detection_state.get("distance_m"))

    raw_distance = _finite_distance(detection_state.get("raw_distance_m"))
    if raw_distance is not None and detection_state.get("depth_quality") in {"good", "weak"}:
        return raw_distance
    return _finite_distance(detection_state.get("distance_m"))


def _robust_recent_median(history):
    values = np.array(list(history)[-DISPLAY_DISTANCE_MEDIAN_WINDOW:], dtype=float)
    if len(values) == 0:
        return None
    if len(values) < 4:
        return float(np.median(values))

    local_median = float(np.median(values))
    residuals = np.abs(values - local_median)
    mad = float(np.median(residuals))
    threshold = max(1.2, 3.0 * 1.4826 * mad)
    filtered = values[residuals <= threshold]
    if len(filtered) == 0:
        filtered = values
    return float(np.median(filtered))


def _limit_display_step(previous, proposed):
    delta = float(proposed) - float(previous)
    if delta >= 0.0:
        max_step = float(DISPLAY_DISTANCE_MAX_RECEDING_STEP_M)
    else:
        max_step = float(DISPLAY_DISTANCE_MAX_APPROACH_STEP_M)
    return float(previous) + float(np.clip(delta, -max_step, max_step))


def _smooth_display_distance(track_id, detection_state, display_distance_state):
    if not DISPLAY_DISTANCE_SMOOTHING_ENABLED:
        return _finite_distance(detection_state.get("distance_m"))

    source = _display_distance_source(detection_state)
    if track_id < 0:
        return source

    state = display_distance_state.get(track_id)
    if state is None:
        state = {
            "history": deque(maxlen=DISPLAY_DISTANCE_HISTORY_LEN),
            "smooth": None,
            "missing_frames": 0,
        }
        display_distance_state[track_id] = state

    if source is None:
        state["missing_frames"] = int(state.get("missing_frames", 0)) + 1
        if state.get("smooth") is not None and state["missing_frames"] <= DISPLAY_DISTANCE_HOLD_FRAMES:
            return float(state["smooth"])
        return None

    state["missing_frames"] = 0
    state["history"].append(float(source))
    target = _robust_recent_median(state["history"])
    if target is None:
        return None

    previous = state.get("smooth")
    if previous is None:
        smooth = float(target)
    else:
        alpha = DISPLAY_DISTANCE_EMA_ALPHA
        if float(target) < float(previous):
            alpha = max(alpha, DISPLAY_DISTANCE_APPROACH_EMA_ALPHA)
        proposed = (1.0 - float(alpha)) * float(previous) + float(alpha) * float(target)
        smooth = _limit_display_step(previous, proposed)

    state["smooth"] = float(smooth)
    return float(smooth)


def _approach_from_history(history, distance_m, timestamp_ms):
    if distance_m is None or not history:
        return 0.0

    history_items = list(history)
    if len(history_items) <= APPROACH_LOOKBACK_FRAMES:
        return 0.0

    old_ts, old_distance = history_items[-1 - APPROACH_LOOKBACK_FRAMES]
    dt = max((timestamp_ms - old_ts) / 1000.0, 1e-6)
    return float(max(0.0, (old_distance - distance_m) / dt))


def _velocity_from_position_history(history):
    if not history:
        return None, None

    history_items = list(history)
    if len(history_items) <= APPROACH_LOOKBACK_FRAMES:
        return None, None

    old_ts, old_x, old_z = history_items[-1 - APPROACH_LOOKBACK_FRAMES]
    new_ts, new_x, new_z = history_items[-1]
    dt = max((new_ts - old_ts) / 1000.0, 1e-6)
    return float((new_x - old_x) / dt), float((new_z - old_z) / dt)


def _smooth_detection_display(obj, display_track_state, frame_w, frame_h):
    track_id = int(obj.get("track_id", -1))
    if track_id < 0:
        return obj

    bbox = obj.get("bbox_xyxy")
    bearing_deg = to_float_or_none(obj.get("bearing_deg"))
    if bbox is None or len(bbox) != 4:
        return obj

    center_x, center_y, width, height = _bbox_xyxy_to_center_size(bbox)
    state = display_track_state.get(track_id, {})

    smooth_center_x = _ema(state.get("center_x"), center_x, BBOX_EMA_ALPHA)
    smooth_center_y = _ema(state.get("center_y"), center_y, BBOX_EMA_ALPHA)
    smooth_width = _ema(state.get("width"), width, BBOX_EMA_ALPHA)
    smooth_height = _ema(state.get("height"), height, BBOX_EMA_ALPHA)
    smooth_bearing = bearing_deg
    if bearing_deg is not None:
        smooth_bearing = _ema(state.get("bearing_deg"), bearing_deg, BEARING_EMA_ALPHA)

    smoothed_bbox = _center_size_to_bbox_xyxy(
        smooth_center_x,
        smooth_center_y,
        smooth_width,
        smooth_height,
        frame_w,
        frame_h,
    )
    if smoothed_bbox is not None:
        obj["bbox_xyxy"] = smoothed_bbox
    if smooth_bearing is not None:
        obj["bearing_deg"] = round(float(smooth_bearing), 2)

    display_track_state[track_id] = {
        "center_x": smooth_center_x,
        "center_y": smooth_center_y,
        "width": smooth_width,
        "height": smooth_height,
        "bearing_deg": smooth_bearing,
    }
    return obj
