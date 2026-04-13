# -*- coding: utf-8 -*-
"""Pure bbox geometry helpers shared across detection, depth, tracking, risk."""

import numpy as np

from ..config import HFOV_DEG


def _clip_bbox_xyxy(bbox_xyxy, frame_w, frame_h):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w - 1))
    y2 = max(0, min(y2, frame_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_xyxy_to_center_size(bbox_xyxy):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        max(1.0, x2 - x1),
        max(1.0, y2 - y1),
    )


def _center_size_to_bbox_xyxy(cx, cy, width, height, frame_w, frame_h):
    half_w = max(0.5, float(width) / 2.0)
    half_h = max(0.5, float(height) / 2.0)
    bbox = [
        cx - half_w,
        cy - half_h,
        cx + half_w,
        cy + half_h,
    ]
    return _clip_bbox_xyxy(bbox, frame_w, frame_h)


def _bbox_bearing_deg(bbox_xyxy, frame_w, hfov_deg=HFOV_DEG):
    x1, _, x2, _ = bbox_xyxy
    center_x = 0.5 * (x1 + x2)
    normalized = ((center_x / float(frame_w)) - 0.5) * 2.0
    bearing = normalized * (hfov_deg / 2.0)
    return float(np.clip(bearing, -hfov_deg / 2.0, hfov_deg / 2.0))


def _bearing_distance_to_xz(distance_m, bearing_deg):
    if distance_m is None or bearing_deg is None:
        return None, None
    rad = float(bearing_deg) * np.pi / 180.0
    x = float(distance_m) * float(np.sin(rad))
    z = float(distance_m) * float(np.cos(rad))
    return x, z
