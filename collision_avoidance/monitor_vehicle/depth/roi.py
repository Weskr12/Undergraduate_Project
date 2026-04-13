# -*- coding: utf-8 -*-
"""Compute a scalar raw depth for a bbox by sampling its bottom-center ROI."""

import numpy as np

from ..config import MIN_DEPTH
from ..detection.bbox import _clip_bbox_xyxy


def compute_object_distance(depth_map, bbox_xyxy):
    if depth_map is None:
        return None

    h, w = depth_map.shape[:2]
    clipped = _clip_bbox_xyxy(bbox_xyxy, w, h)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    bh = y2 - y1

    rx1 = x1 + int(round(0.2 * bw))
    rx2 = x1 + int(round(0.8 * bw))
    ry1 = y1 + int(round(0.7 * bh))
    ry2 = y2

    rx1 = max(0, min(rx1, w - 1))
    rx2 = max(0, min(rx2, w))
    ry1 = max(0, min(ry1, h - 1))
    ry2 = max(0, min(ry2, h))

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = depth_map[ry1:ry2, rx1:rx2]
    valid = roi[np.isfinite(roi) & (roi > MIN_DEPTH)]
    if valid.size == 0:
        return None

    return float(np.percentile(valid, 35))
