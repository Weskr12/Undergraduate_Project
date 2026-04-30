# -*- coding: utf-8 -*-
"""Compute a scalar raw depth for a bbox by sampling its bottom-center ROI."""

import numpy as np

from ..config import (
    DEPTH_SAMPLE_PERCENTILE,
    MAX_DEPTH_ROI_IQR_M,
    MAX_VALID_DEPTH_M,
    MIN_DEPTH,
    MIN_DEPTH_VALID_RATIO,
    MIN_RELIABLE_BBOX_AREA_PX,
    MIN_RELIABLE_BBOX_HEIGHT_PX,
    MIN_RELIABLE_BBOX_WIDTH_PX,
)
from ..detection.bbox import _clip_bbox_xyxy


def compute_object_distance_with_debug(depth_map, bbox_xyxy, percentile=DEPTH_SAMPLE_PERCENTILE):
    percentile = float(percentile)
    sample_method = f"bbox_bottom_center_p{percentile:g}"
    empty_result = {
        "depth_value": None,
        "depth_roi": None,
        "roi_pixel_count": 0,
        "depth_valid_ratio": 0.0,
        "depth_iqr": None,
        "depth_percentile_used": percentile,
        "depth_p25": None,
        "depth_p35": None,
        "depth_p40": None,
        "depth_p50": None,
        "depth_quality": "bad",
        "depth_sample_method": sample_method,
        "bbox_width_px": None,
        "bbox_height_px": None,
        "bbox_area_px": None,
        "bbox_quality_reason": None,
    }

    if depth_map is None:
        return empty_result

    h, w = depth_map.shape[:2]
    clipped = _clip_bbox_xyxy(bbox_xyxy, w, h)
    if clipped is None:
        return empty_result

    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    bh = y2 - y1
    bbox_area = bw * bh
    bbox_quality_reasons = []
    if bw < MIN_RELIABLE_BBOX_WIDTH_PX:
        bbox_quality_reasons.append("bbox_width_too_small")
    if bh < MIN_RELIABLE_BBOX_HEIGHT_PX:
        bbox_quality_reasons.append("bbox_height_too_small")
    if bbox_area < MIN_RELIABLE_BBOX_AREA_PX:
        bbox_quality_reasons.append("bbox_area_too_small")
    bbox_debug = {
        "bbox_width_px": int(bw),
        "bbox_height_px": int(bh),
        "bbox_area_px": int(bbox_area),
        "bbox_quality_reason": None if not bbox_quality_reasons else ",".join(bbox_quality_reasons),
    }

    rx1 = x1 + int(round(0.2 * bw))
    rx2 = x1 + int(round(0.8 * bw))
    ry1 = y1 + int(round(0.7 * bh))
    ry2 = y2

    rx1 = max(0, min(rx1, w - 1))
    rx2 = max(0, min(rx2, w))
    ry1 = max(0, min(ry1, h - 1))
    ry2 = max(0, min(ry2, h))

    if rx2 <= rx1 or ry2 <= ry1:
        return {
            **empty_result,
            **bbox_debug,
            "depth_roi": {"x1": int(rx1), "y1": int(ry1), "x2": int(rx2), "y2": int(ry2)},
        }

    roi = depth_map[ry1:ry2, rx1:rx2]
    valid_mask = np.isfinite(roi) & (roi > MIN_DEPTH) & (roi <= MAX_VALID_DEPTH_M)
    valid = roi[valid_mask]
    total_count = int(roi.size)
    valid_ratio = float(valid.size / max(total_count, 1))
    depth_roi = {"x1": int(rx1), "y1": int(ry1), "x2": int(rx2), "y2": int(ry2)}

    if valid.size == 0:
        return {
            **empty_result,
            **bbox_debug,
            "depth_roi": depth_roi,
            "roi_pixel_count": total_count,
            "depth_valid_ratio": valid_ratio,
        }

    q25, p35, p40, p50, q75 = np.percentile(valid, [25, 35, 40, 50, 75])
    depth_iqr = float(q75 - q25)
    depth_quality = "good"
    if valid_ratio < MIN_DEPTH_VALID_RATIO:
        depth_quality = "bad"
    elif depth_iqr > MAX_DEPTH_ROI_IQR_M:
        depth_quality = "weak"
    if bbox_quality_reasons:
        depth_quality = "bad"

    return {
        "depth_value": float(np.percentile(valid, percentile)),
        **bbox_debug,
        "depth_roi": depth_roi,
        "roi_pixel_count": total_count,
        "depth_valid_ratio": valid_ratio,
        "depth_iqr": depth_iqr,
        "depth_percentile_used": percentile,
        "depth_p25": float(q25),
        "depth_p35": float(p35),
        "depth_p40": float(p40),
        "depth_p50": float(p50),
        "depth_quality": depth_quality,
        "depth_sample_method": sample_method,
    }


def compute_object_distance(depth_map, bbox_xyxy):
    return compute_object_distance_with_debug(depth_map, bbox_xyxy)["depth_value"]
