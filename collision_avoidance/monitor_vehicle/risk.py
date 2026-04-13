# -*- coding: utf-8 -*-
"""Risk assessment: distance bands, TTC, risk levels, depth priority scoring."""

import math

import numpy as np

from .config import (
    DEPTH_PRIORITY_MAX_TARGETS,
    HFOV_DEG,
    HOOK_TURN_CLASS_ID,
    TTC_CRITICAL_SECONDS,
    TTC_HIGH_SECONDS,
    TTC_MAX_SECONDS,
    TTC_MEDIUM_SECONDS,
    TTC_MIN_APPROACH_MPS,
    TTC_STABLE_BEARING_STD_DEG,
    TTC_STABLE_CENTER_STD_NORM,
    TTC_STABLE_MIN_FRAMES,
)
from .detection.bbox import _bbox_bearing_deg
from .utils import to_float_or_none


def _distance_band(distance_m):
    if distance_m is None:
        return "unknown"
    if distance_m <= 8.0:
        return "near"
    if distance_m <= 15.0:
        return "mid"
    return "far"


def _risk_level(distance_m, approach_mps):
    if distance_m is None:
        return "unknown"
    if distance_m <= 4.0:
        return "critical"
    if distance_m <= 8.0:
        return "high"
    if distance_m <= 15.0:
        return "medium"
    return "low"


def _compute_ttc_s(distance_m, approach_mps):
    if distance_m is None:
        return None

    distance_m = to_float_or_none(distance_m)
    approach_mps = to_float_or_none(approach_mps)
    if distance_m is None or approach_mps is None or approach_mps <= TTC_MIN_APPROACH_MPS:
        return None

    ttc_s = distance_m / max(approach_mps, 1e-6)
    if not math.isfinite(ttc_s) or ttc_s > TTC_MAX_SECONDS:
        return None
    return float(max(0.0, ttc_s))


def _risk_level_from_ttc(distance_m, approach_mps, ttc_s):
    if distance_m is None:
        return "unknown"
    if distance_m <= 4.0:
        return "critical"
    if ttc_s is not None:
        if ttc_s <= TTC_CRITICAL_SECONDS:
            return "critical"
        if ttc_s <= TTC_HIGH_SECONDS:
            return "high"
        if ttc_s <= TTC_MEDIUM_SECONDS:
            return "medium"
        return "low"
    return _risk_level(distance_m, approach_mps)


def _is_stable_for_ttc(track_observations):
    if track_observations is None or len(track_observations) < TTC_STABLE_MIN_FRAMES:
        return False

    center_x_norm = np.array([item["center_x_norm"] for item in track_observations], dtype=np.float32)
    bearing_deg = np.array([item["bearing_deg"] for item in track_observations], dtype=np.float32)

    center_std = float(np.std(center_x_norm))
    bearing_std = float(np.std(bearing_deg))
    return center_std <= TTC_STABLE_CENTER_STD_NORM and bearing_std <= TTC_STABLE_BEARING_STD_DEG


def _depth_priority_score(det, frame_w, frame_h):
    bbox = det.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        return float("-inf")

    x1, y1, x2, y2 = [float(v) for v in bbox]
    bbox_area = max(0.0, (x2 - x1) * max(0.0, y2 - y1))
    area_ratio = bbox_area / max(float(frame_w * frame_h), 1.0)
    bottom_ratio = np.clip(y2 / max(float(frame_h), 1.0), 0.0, 1.0)
    bearing = abs(_bbox_bearing_deg(bbox, frame_w))
    center_score = 1.0 - np.clip(bearing / max(HFOV_DEG / 2.0, 1e-6), 0.0, 1.0)
    return float(center_score * 3.0 + bottom_ratio * 1.5 + area_ratio * 8.0)


def _select_depth_target_indexes(detections, frame_w, frame_h, max_targets=DEPTH_PRIORITY_MAX_TARGETS):
    if max_targets <= 0:
        return set()

    scored_targets = []
    for det_idx, det in enumerate(detections):
        if int(det.get("class_id", -1)) == HOOK_TURN_CLASS_ID:
            continue
        score = _depth_priority_score(det, frame_w, frame_h)
        if not math.isfinite(score):
            continue
        scored_targets.append((score, det_idx))

    scored_targets.sort(reverse=True)
    return {det_idx for _, det_idx in scored_targets[:max_targets]}
