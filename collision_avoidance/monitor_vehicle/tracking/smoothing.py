# -*- coding: utf-8 -*-
"""Per-track distance smoothing, approach velocity, and display EMA."""

import numpy as np

from ..config import (
    APPROACH_LOOKBACK_FRAMES,
    BBOX_EMA_ALPHA,
    BEARING_EMA_ALPHA,
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
