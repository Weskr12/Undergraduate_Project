# -*- coding: utf-8 -*-
"""Main run_mvp_pipeline orchestration and its frame-level helpers."""

import gc
import math
import time
from copy import deepcopy
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from .config import (
    DEPTH_MAX_ABS_JUMP_M,
    DEPTH_MAX_PLAUSIBLE_SPEED_MPS,
    DEPTH_MAX_REL_JUMP,
    DEPTH_INFER_EVERY_N,
    DEPTH_DISPLAY_MAX_M,
    DEPTH_MIN_CONFIDENCE_FOR_BASELINE,
    DEPTH_NEAR_LOCK_RECOVERY_MAX_M,
    DEPTH_NEAR_LOCK_RECOVERY_RAW_MIN_M,
    DEPTH_PRO_CPU_INFER_EVERY_N,
    DEPTH_RECOVERY_MIN_CONFIDENCE,
    DEPTH_RECOVERY_MIN_CONSECUTIVE,
    DEPTH_RELIABLE_MAX_M,
    DETECTION_HOLD_FRAMES,
    DRAW_HELD_DETECTIONS,
    FAR_DISTANCE_CLAMP_M,
    HFOV_DEG,
    HOOK_TURN_CLASS_ID,
    MAX_VALID_DEPTH_M,
    MIN_DEPTH,
    MIN_RELIABLE_BBOX_AREA_PX,
    YOLO_FAR_DEPTH_CONFIDENCE,
    YOLO_FAR_DISTANCE_M,
    YOLO_FAR_FALLBACK_ENABLED,
    YOLO_FAR_MAX_BBOX_AREA_PX,
    YOLO_FAR_MAX_BBOX_HEIGHT_PX,
    YOLO_FAR_MAX_BBOX_WIDTH_PX,
    YOLO_FAR_MIN_CONFIDENCE,
    YOLO_FAR_UNRELIABLE_MAX_BBOX_AREA_PX,
    YOLO_FAR_UNRELIABLE_MAX_BBOX_HEIGHT_PX,
    YOLO_FAR_UNRELIABLE_MAX_BBOX_WIDTH_PX,
    YOLO_FAR_UNRELIABLE_RAW_MIN_M,
)
from .depth.backends import infer_depth_map, load_depth_model
from .depth.calibration import load_calibration, raw_depth_to_distance_m
from .depth.roi import compute_object_distance_with_debug
from .detection.bbox import _bearing_distance_to_xz
from .detection.yolo_tracker import _detect_with_tracking
from .output.annotations import _draw_detection, _draw_frame_summary
from .output.payload import (
    _append_timeline_jsonl,
    _append_birdseye_timeline_jsonl,
    _build_output_object,
    _update_payload_summary,
    build_radar_payload,
    write_json_file,
    write_pretty_json_from_jsonl,
    write_birdseye_live_json,
    write_live_json,
)
from .output.radar_panel import (
    _compose_output_frame,
    _compute_output_video_size,
    _resize_output_frame,
)
from .paths import RUN_ID, YOLO_MODEL_PATH
from .risk import (
    _compute_ttc_s,
    _distance_band,
    _distance_band_with_hysteresis,
    _risk_level,
    _risk_level_from_ttc,
    _select_depth_target_indexes,
)
from .runtime import _resolve_runtime_settings
from .tracking.smoothing import (
    _approach_from_history,
    _smooth_detection_display,
    _smoothed_distance_from_history,
    _velocity_from_position_history,
)
from .tracking.state import (
    _cleanup_stale_tracks,
    _create_pipeline_tracking_state,
    _update_track_stability,
)
from .utils import ensure_parent


def _open_pipeline_video_io(video_path, output_video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_size = _compute_output_video_size(frame_w, frame_h)

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_video_size,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_video_path}")

    return {
        "cap": cap,
        "writer": writer,
        "fps": fps,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "output_video_size": output_video_size,
    }


def _load_pipeline_models(calib_path):
    runtime = _resolve_runtime_settings()
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    if hasattr(yolo_model, "to"):
        try:
            yolo_model.to(runtime["device_str"])
        except Exception:
            pass

    depth_state = load_depth_model(runtime=runtime)
    calib_model = None
    if depth_state.get("units") != "metric_m":
        calib_model = load_calibration(calib_path)

    depth_infer_every_n = DEPTH_INFER_EVERY_N
    if depth_state.get("backend") == "depth_pro" and runtime["device_type"] == "cpu":
        depth_infer_every_n = max(DEPTH_INFER_EVERY_N, DEPTH_PRO_CPU_INFER_EVERY_N)

    return {
        "runtime": runtime,
        "yolo_model": yolo_model,
        "depth_state": depth_state,
        "calib_model": calib_model,
        "depth_infer_every_n": depth_infer_every_n,
    }


def _infer_frame_depth_if_needed(
    frame,
    frame_idx,
    detections,
    frame_w,
    frame_h,
    depth_infer_every_n,
    depth_state,
    tracking_state,
):
    depth_target_indexes = _select_depth_target_indexes(detections, frame_w, frame_h)
    should_run_depth = bool(depth_target_indexes) and (
        (frame_idx % depth_infer_every_n == 0) or (tracking_state["last_depth_map"] is None)
    )
    frame_depth_stale = bool(depth_target_indexes) and not should_run_depth

    if should_run_depth:
        try:
            tracking_state["last_depth_map"] = infer_depth_map(frame, depth_state)
            frame_depth_stale = False
        except Exception:
            frame_depth_stale = True

    return depth_target_indexes, frame_depth_stale


def _empty_depth_debug(depth_source="unknown", depth_quality="bad"):
    return {
        "depth_raw_m": None,
        "depth_smooth_m": None,
        "depth_used_m": None,
        "raw_distance_m": None,
        "distance_after_gate_m": None,
        "distance_smooth_m": None,
        "depth_source": depth_source,
        "depth_quality": depth_quality,
        "depth_valid_ratio": None,
        "depth_roi": None,
        "roi_xyxy": None,
        "roi_pixel_count": 0,
        "depth_iqr": None,
        "depth_percentile_used": None,
        "depth_p25": None,
        "depth_p35": None,
        "depth_p40": None,
        "depth_p50": None,
        "depth_sample_method": None,
        "bbox_width_px": None,
        "bbox_height_px": None,
        "bbox_area_px": None,
        "bbox_area": None,
        "depth_reliable": False,
        "depth_confidence": 0.0,
        "confidence_scale": 1.0,
        "unity_z_raw": None,
        "unity_z_used": None,
        "is_outlier": False,
        "outlier_reason": None,
        "gate_decision": "missing",
        "gate_reason": None,
        "missing_frames": 0,
        "missed_frames": 0,
        "distance_source": "missing",
    }


def _join_reasons(*reason_groups):
    reasons = []
    for group in reason_groups:
        if not group:
            continue
        if isinstance(group, str):
            reasons.extend(part for part in group.split(",") if part)
        else:
            reasons.extend(str(part) for part in group if part)
    deduped = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return None if not deduped else ",".join(deduped)


def _depth_confidence_from_debug(raw_depth_m, gate_quality, roi_debug):
    quality = str(gate_quality or "bad")
    if quality == "good":
        confidence = 1.0
    elif quality == "weak":
        confidence = 0.65
    else:
        confidence = 0.25

    valid_ratio = roi_debug.get("depth_valid_ratio")
    if valid_ratio is not None:
        confidence *= 0.5 + 0.5 * max(0.0, min(float(valid_ratio), 1.0))

    bbox_area = roi_debug.get("bbox_area_px")
    if bbox_area is not None:
        if int(bbox_area) < MIN_RELIABLE_BBOX_AREA_PX:
            confidence = min(confidence, 0.25)
        elif int(bbox_area) < MIN_RELIABLE_BBOX_AREA_PX * 3:
            confidence *= 0.85

    if raw_depth_m is not None and math.isfinite(float(raw_depth_m)):
        if raw_depth_m > DEPTH_DISPLAY_MAX_M:
            confidence = min(confidence, 0.25)
        elif raw_depth_m > DEPTH_RELIABLE_MAX_M:
            confidence = min(confidence, 0.45)

    return float(max(0.0, min(confidence, 1.0)))


def _risk_level_with_confidence(distance_m, approach_mps, depth_confidence, depth_reliable):
    if distance_m is None:
        return "unknown"
    if depth_reliable:
        return _risk_level(distance_m, approach_mps)
    if depth_confidence < DEPTH_MIN_CONFIDENCE_FOR_BASELINE:
        if distance_m <= 8.0:
            return "close_unknown"
        if distance_m <= 15.0:
            return "medium_uncertain"
        return "unknown"
    return _risk_level(distance_m, approach_mps)


def _bbox_size_debug_from_xyxy(bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width, height, width * height


def _is_yolo_far_candidate(bbox, det_confidence):
    if not YOLO_FAR_FALLBACK_ENABLED:
        return False
    if det_confidence is None or float(det_confidence) < YOLO_FAR_MIN_CONFIDENCE:
        return False

    width, height, area = _bbox_size_debug_from_xyxy(bbox)
    return (
        width <= YOLO_FAR_MAX_BBOX_WIDTH_PX
        or height <= YOLO_FAR_MAX_BBOX_HEIGHT_PX
        or area <= YOLO_FAR_MAX_BBOX_AREA_PX
    )


def _is_yolo_unreliable_depth_far_candidate(bbox, det_confidence, raw_distance_m):
    if not YOLO_FAR_FALLBACK_ENABLED:
        return False
    if det_confidence is None or float(det_confidence) < YOLO_FAR_MIN_CONFIDENCE:
        return False
    if raw_distance_m is None or not math.isfinite(float(raw_distance_m)):
        return False
    if float(raw_distance_m) < YOLO_FAR_UNRELIABLE_RAW_MIN_M:
        return False

    width, height, area = _bbox_size_debug_from_xyxy(bbox)
    return (
        width <= YOLO_FAR_UNRELIABLE_MAX_BBOX_WIDTH_PX
        or height <= YOLO_FAR_UNRELIABLE_MAX_BBOX_HEIGHT_PX
        or area <= YOLO_FAR_UNRELIABLE_MAX_BBOX_AREA_PX
    )


def _should_apply_yolo_far_fallback(bbox, det_confidence, distance_m, depth_debug):
    raw_distance_m = depth_debug.get("raw_distance_m", depth_debug.get("depth_raw_m"))
    far_candidate = _is_yolo_far_candidate(bbox, det_confidence)
    unreliable_depth_far_candidate = _is_yolo_unreliable_depth_far_candidate(
        bbox,
        det_confidence,
        raw_distance_m,
    )
    if not (far_candidate or unreliable_depth_far_candidate):
        return False
    if distance_m is None:
        return True

    depth_quality = str(depth_debug.get("depth_quality", "bad"))
    distance_source = str(depth_debug.get("distance_source", "missing"))
    depth_confidence = depth_debug.get("depth_confidence")
    low_confidence = (
        depth_confidence is None
        or float(depth_confidence) < DEPTH_MIN_CONFIDENCE_FOR_BASELINE
    )
    unreliable_depth = (
        depth_quality in {"bad", "weak"}
        or bool(depth_debug.get("is_outlier", False))
        or low_confidence
    )
    return unreliable_depth and distance_source in {"held_previous", "missing"}


def _append_rejected_depth_candidate(track_id, timestamp_ms, raw_depth_m, depth_confidence, bbox_area, tracking_state):
    candidates = tracking_state["rejected_depth_candidates"][track_id]
    candidates.append(
        {
            "timestamp_ms": int(timestamp_ms),
            "raw_depth_m": float(raw_depth_m),
            "depth_confidence": float(depth_confidence),
            "bbox_area": int(bbox_area or 0),
        }
    )
    return candidates


def _should_recover_from_consistent_approach(candidates, previous_smooth_m):
    if previous_smooth_m is None or len(candidates) < DEPTH_RECOVERY_MIN_CONSECUTIVE:
        return False

    recent = list(candidates)[-DEPTH_RECOVERY_MIN_CONSECUTIVE:]
    raw_values = [item["raw_depth_m"] for item in recent]
    if not all(math.isfinite(value) for value in raw_values):
        return False
    if not all(item["depth_confidence"] >= DEPTH_RECOVERY_MIN_CONFIDENCE for item in recent):
        return False
    if not all(item["bbox_area"] >= MIN_RELIABLE_BBOX_AREA_PX for item in recent):
        return False
    if not all(value < float(previous_smooth_m) for value in raw_values):
        return False

    decreasing_or_stable = all(
        raw_values[idx] <= raw_values[idx - 1] + 1.0
        for idx in range(1, len(raw_values))
    )
    return bool(decreasing_or_stable)


def should_accept_depth_sample(
    track_id,
    timestamp_ms,
    raw_depth_m,
    previous_smooth_m,
    previous_history,
    depth_quality=None,
):
    reasons = []
    implied_speed_mps = None

    if depth_quality == "bad":
        reasons.append("roi_quality_bad")

    if raw_depth_m is None or not math.isfinite(float(raw_depth_m)):
        reasons.append("invalid_depth")
    elif raw_depth_m <= MIN_DEPTH:
        reasons.append("non_positive_depth")
    elif raw_depth_m > MAX_VALID_DEPTH_M:
        reasons.append("depth_over_max")

    if previous_smooth_m is not None and raw_depth_m is not None and math.isfinite(float(raw_depth_m)):
        jump_m = abs(float(raw_depth_m) - float(previous_smooth_m))
        near_lock_recovery = (
            depth_quality != "bad"
            and previous_smooth_m <= DEPTH_NEAR_LOCK_RECOVERY_MAX_M
            and raw_depth_m >= DEPTH_NEAR_LOCK_RECOVERY_RAW_MIN_M
            and raw_depth_m <= DEPTH_RELIABLE_MAX_M
        )
        if not near_lock_recovery:
            if jump_m > DEPTH_MAX_ABS_JUMP_M:
                reasons.append("abs_jump")

            if previous_smooth_m > MIN_DEPTH:
                rel_jump = jump_m / max(float(previous_smooth_m), MIN_DEPTH)
                if rel_jump > DEPTH_MAX_REL_JUMP:
                    reasons.append("rel_jump")

            if previous_history:
                history_items = list(previous_history)
                previous_ts = int(history_items[-1][0])
                dt_s = max((int(timestamp_ms) - previous_ts) / 1000.0, 1e-6)
                implied_speed_mps = jump_m / dt_s
                if implied_speed_mps > DEPTH_MAX_PLAUSIBLE_SPEED_MPS:
                    reasons.append("implied_speed")

    return {
        "accepted": not reasons,
        "is_outlier": bool(reasons),
        "outlier_reason": None if not reasons else ",".join(reasons),
        "implied_speed_mps": implied_speed_mps,
        "near_lock_recovery": bool(
            previous_smooth_m is not None
            and raw_depth_m is not None
            and math.isfinite(float(raw_depth_m))
            and depth_quality != "bad"
            and previous_smooth_m <= DEPTH_NEAR_LOCK_RECOVERY_MAX_M
            and raw_depth_m >= DEPTH_NEAR_LOCK_RECOVERY_RAW_MIN_M
            and raw_depth_m <= DEPTH_RELIABLE_MAX_M
        ),
        "track_id": int(track_id),
    }


def _depth_used_for_position(distance_m, distance_band):
    if distance_m is None:
        return None
    if distance_band == "far":
        return min(float(distance_m), FAR_DISTANCE_CLAMP_M)
    return float(distance_m)


def _position_dict_from_xz(pos_x, pos_z):
    if pos_x is None or pos_z is None:
        return None
    return {
        "x": round(float(pos_x), 3),
        "z": round(float(pos_z), 3),
    }


def _velocity_dict_from_xz(vel_x, vel_z):
    if vel_x is None or vel_z is None:
        return None
    return {
        "vx": round(float(vel_x), 3),
        "vz": round(float(vel_z), 3),
    }


def _estimate_detection_state(
    det_idx,
    track_id,
    bbox,
    det_confidence,
    timestamp_ms,
    is_hook_turn,
    stable_for_ttc,
    depth_target_indexes,
    frame_depth_stale,
    depth_state,
    calib_model,
    tracking_state,
):
    raw_depth = None
    raw_depth_m = None
    distance_m = None
    approach_mps = 0.0
    ttc_s = None
    distance_band = "unknown"
    risk_level = "unknown"
    det_depth_stale = False
    depth_debug = _empty_depth_debug()

    if is_hook_turn:
        return {
            "raw_depth": raw_depth,
            "distance_m": distance_m,
            "approach_mps": approach_mps,
            "ttc_s": ttc_s,
            "distance_band": distance_band,
            "risk_level": "low",
            "det_depth_stale": det_depth_stale,
            **depth_debug,
        }

    history = tracking_state["distance_history"][track_id] if track_id >= 0 else None
    previous_smooth_m = _smoothed_distance_from_history(history)
    should_estimate_distance = (det_idx in depth_target_indexes) and (tracking_state["last_depth_map"] is not None)

    if should_estimate_distance:
        roi_debug = compute_object_distance_with_debug(tracking_state["last_depth_map"], bbox)
        raw_depth = roi_debug["depth_value"]
        raw_depth_m = raw_depth_to_distance_m(raw_depth, depth_state, calib_model)
        depth_source = "reused_depth_map" if frame_depth_stale else "new_depth_map"
        depth_debug.update(
            {
                "depth_raw_m": raw_depth_m,
                "raw_distance_m": raw_depth_m,
                "depth_source": depth_source,
                "depth_quality": roi_debug["depth_quality"],
                "depth_valid_ratio": roi_debug["depth_valid_ratio"],
                "depth_roi": roi_debug["depth_roi"],
                "roi_xyxy": roi_debug["depth_roi"],
                "roi_pixel_count": roi_debug["roi_pixel_count"],
                "depth_iqr": roi_debug["depth_iqr"],
                "depth_percentile_used": roi_debug["depth_percentile_used"],
                "depth_p25": roi_debug["depth_p25"],
                "depth_p35": roi_debug["depth_p35"],
                "depth_p40": roi_debug["depth_p40"],
                "depth_p50": roi_debug["depth_p50"],
                "depth_sample_method": roi_debug["depth_sample_method"],
                "bbox_width_px": roi_debug["bbox_width_px"],
                "bbox_height_px": roi_debug["bbox_height_px"],
                "bbox_area_px": roi_debug["bbox_area_px"],
                "bbox_area": roi_debug["bbox_area_px"],
            }
        )
        gate_quality = roi_debug["depth_quality"]
        if gate_quality == "bad":
            depth_debug["confidence_scale"] = 0.3
        elif gate_quality == "weak":
            depth_debug["confidence_scale"] = 0.6
        range_reason = None
        if raw_depth_m is not None:
            if raw_depth_m > DEPTH_DISPLAY_MAX_M:
                gate_quality = "bad"
                range_reason = "depth_beyond_reliable_range"
                depth_debug["confidence_scale"] = 0.3
            elif raw_depth_m > DEPTH_RELIABLE_MAX_M and gate_quality == "good":
                gate_quality = "weak"
                depth_debug["confidence_scale"] = 0.6
        depth_debug["depth_quality"] = gate_quality
        depth_confidence = _depth_confidence_from_debug(raw_depth_m, gate_quality, roi_debug)
        depth_debug["depth_confidence"] = depth_confidence

        if history is not None:
            gate = should_accept_depth_sample(
                track_id=track_id,
                timestamp_ms=timestamp_ms,
                raw_depth_m=raw_depth_m,
                previous_smooth_m=previous_smooth_m,
                previous_history=history,
                depth_quality=gate_quality,
            )
            bbox_reason = "bbox_too_small_for_reliable_depth" if roi_debug.get("bbox_quality_reason") else None
            if bbox_reason:
                gate["accepted"] = False
                gate["is_outlier"] = True
            if range_reason:
                gate["accepted"] = False
                gate["is_outlier"] = True
            confidence_reason = None
            if depth_confidence < DEPTH_MIN_CONFIDENCE_FOR_BASELINE:
                confidence_reason = "low_confidence_baseline"
                gate["accepted"] = False
                gate["is_outlier"] = True
            recovered = False
            if (
                (not gate["accepted"])
                and raw_depth_m is not None
                and math.isfinite(float(raw_depth_m))
                and gate_quality == "good"
                and depth_confidence >= DEPTH_RECOVERY_MIN_CONFIDENCE
            ):
                candidates = _append_rejected_depth_candidate(
                    track_id=track_id,
                    timestamp_ms=timestamp_ms,
                    raw_depth_m=raw_depth_m,
                    depth_confidence=depth_confidence,
                    bbox_area=roi_debug.get("bbox_area_px"),
                    tracking_state=tracking_state,
                )
                recovered = _should_recover_from_consistent_approach(candidates, previous_smooth_m)
                if recovered:
                    gate["accepted"] = True
                    gate["is_outlier"] = False
                    gate["outlier_reason"] = None
            else:
                tracking_state["rejected_depth_candidates"][track_id].clear()
            depth_debug["is_outlier"] = gate["is_outlier"]
            depth_debug["outlier_reason"] = _join_reasons(
                gate["outlier_reason"],
                bbox_reason,
                range_reason,
                confidence_reason,
            )
            if gate["accepted"]:
                if gate.get("near_lock_recovery") or recovered:
                    history.clear()
                    tracking_state["distance_confidence_history"][track_id].clear()
                history.append((timestamp_ms, float(raw_depth_m)))
                tracking_state["distance_confidence_history"][track_id].append(
                    (timestamp_ms, float(depth_confidence))
                )
                tracking_state["rejected_depth_candidates"][track_id].clear()
                depth_debug["gate_decision"] = "accepted"
                depth_debug["gate_reason"] = "recovery_consistent_approach" if recovered else None
                depth_debug["distance_after_gate_m"] = raw_depth_m
                depth_debug["distance_source"] = "measured"
            else:
                depth_debug["gate_decision"] = "held_previous" if previous_smooth_m is not None else "rejected"
                depth_debug["gate_reason"] = depth_debug["outlier_reason"]
                depth_debug["distance_after_gate_m"] = previous_smooth_m
                depth_debug["distance_source"] = "held_previous" if previous_smooth_m is not None else "missing"
            distance_m = _smoothed_distance_from_history(history)
            if (
                distance_m is None
                and range_reason == "depth_beyond_reliable_range"
                and depth_confidence >= DEPTH_MIN_CONFIDENCE_FOR_BASELINE
            ):
                distance_m = FAR_DISTANCE_CLAMP_M
            approach_mps = _approach_from_history(history, distance_m, timestamp_ms)
        else:
            distance_m = raw_depth_m
            depth_debug["depth_smooth_m"] = raw_depth_m
            depth_debug["distance_after_gate_m"] = raw_depth_m
            depth_debug["gate_decision"] = "accepted"
            depth_debug["distance_source"] = "measured"

        det_depth_stale = bool(frame_depth_stale)
    else:
        distance_m = previous_smooth_m
        approach_mps = _approach_from_history(history, distance_m, timestamp_ms)
        det_depth_stale = bool(frame_depth_stale or history or (det_idx not in depth_target_indexes))
        if distance_m is not None:
            confidence_history = tracking_state["distance_confidence_history"][track_id] if track_id >= 0 else None
            previous_confidence = (
                list(confidence_history)[-1][1]
                if confidence_history
                else DEPTH_MIN_CONFIDENCE_FOR_BASELINE
            )
            depth_debug.update(
                {
                    "depth_source": "history_hold",
                    "depth_quality": "weak",
                    "depth_smooth_m": distance_m,
                    "depth_used_m": distance_m,
                    "depth_confidence": previous_confidence,
                    "distance_after_gate_m": distance_m,
                    "distance_source": "held_previous",
                    "gate_decision": "held_previous",
                    "gate_reason": "depth_not_selected_or_stale",
                }
            )

    if (
        not is_hook_turn
        and _should_apply_yolo_far_fallback(bbox, det_confidence, distance_m, depth_debug)
    ):
        bbox_width, bbox_height, bbox_area = _bbox_size_debug_from_xyxy(bbox)
        distance_m = float(YOLO_FAR_DISTANCE_M)
        approach_mps = 0.0
        det_depth_stale = True
        depth_debug.update(
            {
                "depth_source": "yolo_size_fallback",
                "depth_quality": "weak",
                "depth_confidence": float(YOLO_FAR_DEPTH_CONFIDENCE),
                "depth_smooth_m": distance_m,
                "depth_used_m": distance_m,
                "distance_after_gate_m": distance_m,
                "distance_smooth_m": distance_m,
                "distance_source": "yolo_size_fallback",
                "gate_decision": "accepted",
                "gate_reason": "yolo_small_bbox_far_fallback",
                "is_outlier": False,
                "outlier_reason": None,
                "bbox_width_px": depth_debug.get("bbox_width_px") or int(bbox_width),
                "bbox_height_px": depth_debug.get("bbox_height_px") or int(bbox_height),
                "bbox_area_px": depth_debug.get("bbox_area_px") or int(bbox_area),
                "bbox_area": depth_debug.get("bbox_area") or int(bbox_area),
            }
        )

    if track_id >= 0:
        previous_band = tracking_state["distance_band_state"].get(track_id)
        distance_band = _distance_band_with_hysteresis(distance_m, previous_band)
        if distance_band != "unknown":
            tracking_state["distance_band_state"][track_id] = distance_band
    else:
        distance_band = _distance_band(distance_m)

    depth_used_m = _depth_used_for_position(distance_m, distance_band)
    raw_unity_x, raw_unity_z = _bearing_distance_to_xz(raw_depth_m, 0.0)
    _, used_unity_z = _bearing_distance_to_xz(depth_used_m, 0.0)
    depth_reliable = (
        depth_debug["depth_quality"] == "good"
        and distance_m is not None
        and distance_m <= DEPTH_RELIABLE_MAX_M
        and distance_band in {"near", "mid"}
    )
    depth_debug["depth_smooth_m"] = distance_m
    depth_debug["distance_smooth_m"] = distance_m
    depth_debug["depth_used_m"] = depth_used_m
    depth_debug["depth_reliable"] = bool(depth_reliable)
    depth_debug["unity_z_raw"] = raw_unity_z
    depth_debug["unity_z_used"] = used_unity_z

    if stable_for_ttc and depth_reliable:
        ttc_s = _compute_ttc_s(distance_m, approach_mps)
        risk_level = _risk_level_from_ttc(distance_m, approach_mps, ttc_s)
    else:
        risk_level = _risk_level_with_confidence(
            distance_m,
            approach_mps,
            depth_debug["depth_confidence"],
            depth_reliable,
        )

    return {
        "raw_depth": raw_depth,
        "distance_m": distance_m,
        "approach_mps": approach_mps,
        "ttc_s": ttc_s,
        "distance_band": distance_band,
        "risk_level": risk_level,
        "det_depth_stale": det_depth_stale,
        **depth_debug,
    }


def _process_frame_detections(
    frame,
    detections,
    frame_idx,
    timestamp_ms,
    frame_w,
    frame_h,
    depth_state,
    calib_model,
    tracking_state,
    depth_infer_every_n,
):
    depth_target_indexes, frame_depth_stale = _infer_frame_depth_if_needed(
        frame=frame,
        frame_idx=frame_idx,
        detections=detections,
        frame_w=frame_w,
        frame_h=frame_h,
        depth_infer_every_n=depth_infer_every_n,
        depth_state=depth_state,
        tracking_state=tracking_state,
    )

    objects = []
    hook_turn_detected = False
    detected_objects = 0
    valid_distance_objects = 0

    for det_idx, det in enumerate(detections):
        track_id = int(det["track_id"])
        class_id = int(det["class_id"])
        bbox = [int(v) for v in det["bbox_xyxy"]]
        is_hook_turn = class_id == HOOK_TURN_CLASS_ID
        hook_turn_detected = hook_turn_detected or is_hook_turn

        bearing_deg, stable_for_ttc = _update_track_stability(
            track_id=track_id,
            is_hook_turn=is_hook_turn,
            bbox=bbox,
            frame_w=frame_w,
            tracking_state=tracking_state,
        )
        detection_state = _estimate_detection_state(
            det_idx=det_idx,
            track_id=track_id,
            bbox=bbox,
            det_confidence=float(det.get("confidence", 0.0)),
            timestamp_ms=timestamp_ms,
            is_hook_turn=is_hook_turn,
            stable_for_ttc=stable_for_ttc,
            depth_target_indexes=depth_target_indexes,
            frame_depth_stale=frame_depth_stale,
            depth_state=depth_state,
            calib_model=calib_model,
            tracking_state=tracking_state,
        )

        track_status = "active" if track_id in tracking_state["last_seen_frame"] else "new"

        if track_id >= 0:
            tracking_state["track_hit_count"][track_id] += 1
            tracking_state["last_seen_frame"][track_id] = frame_idx
        hit_count = int(tracking_state["track_hit_count"].get(track_id, 0)) if track_id >= 0 else 0
        if track_id < 0:
            track_state = "temporary"
        elif hit_count >= 2:
            track_state = "confirmed"
        else:
            track_state = "tentative"

        if not is_hook_turn:
            detected_objects += 1
            if detection_state["distance_m"] is not None:
                valid_distance_objects += 1

        obj = _build_output_object(
            det=det,
            bbox=bbox,
            bearing_deg=bearing_deg,
            stable_for_ttc=stable_for_ttc,
            detection_state=detection_state,
            is_hook_turn=is_hook_turn,
            position_xz=(None, None),
            velocity_xz=(None, None),
            track_status=track_status,
        )
        obj["track_confidence"] = round(float(det.get("confidence", 0.0)), 4)
        obj["hit_count"] = hit_count
        obj["track_state"] = track_state
        obj["missed_frames"] = 0
        _smooth_detection_display(obj, tracking_state["display_track_state"], frame_w, frame_h)
        pos_x, pos_z = _bearing_distance_to_xz(obj.get("depth_used_m"), obj.get("bearing_deg"))
        velocity_xz = (None, None)
        if track_id >= 0 and pos_x is not None and pos_z is not None:
            pos_history = tracking_state["position_xz_history"][track_id]
            pos_history.append((timestamp_ms, float(pos_x), float(pos_z)))
            velocity_xz = _velocity_from_position_history(pos_history)
        obj["position_m"] = _position_dict_from_xz(pos_x, pos_z)
        obj["velocity_mps"] = _velocity_dict_from_xz(*velocity_xz)
        objects.append(obj)
        _draw_detection(frame, obj)

    return {
        "objects": objects,
        "hook_turn_detected": hook_turn_detected,
        "depth_target_indexes": depth_target_indexes,
        "detected_objects": detected_objects,
        "valid_distance_objects": valid_distance_objects,
    }


def _apply_detection_hold(frame, frame_idx, objects, tracking_state):
    if DETECTION_HOLD_FRAMES <= 0:
        return objects

    current_track_ids = set()
    for obj in objects:
        track_id = int(obj.get("track_id", -1))
        if track_id < 0:
            continue

        current_track_ids.add(track_id)
        tracking_state["last_output_objects"][track_id] = deepcopy(obj)

    held_objects = []
    for track_id, last_seen_frame in list(tracking_state["last_seen_frame"].items()):
        if track_id in current_track_ids:
            continue

        missing_frames = frame_idx - int(last_seen_frame)
        if missing_frames <= 0 or missing_frames > DETECTION_HOLD_FRAMES:
            continue

        previous_obj = tracking_state["last_output_objects"].get(track_id)
        if previous_obj is None or previous_obj.get("hook_turn_detected"):
            continue

        held_obj = deepcopy(previous_obj)
        held_obj["track_status"] = "held"
        held_obj["depth_stale"] = True
        held_obj["depth_source"] = "history_hold"
        held_obj["depth_quality"] = "weak"
        held_obj["held_for_frames"] = int(missing_frames)
        held_obj["missing_frames"] = int(missing_frames)
        held_obj["missed_frames"] = int(missing_frames)
        held_obj["track_state"] = "held"
        held_obj["gate_decision"] = "missing"
        held_obj["gate_reason"] = "track_missing_hold_previous"
        held_obj["distance_source"] = "held_previous"
        held_obj["distance_after_gate_m"] = held_obj.get("distance_m")
        held_obj["distance_smooth_m"] = held_obj.get("distance_m")
        held_obj["approach_mps"] = 0.0
        held_obj["ttc_s"] = None
        held_objects.append(held_obj)
        if DRAW_HELD_DETECTIONS:
            _draw_detection(frame, held_obj)

    if held_objects:
        return objects + held_objects

    return objects


def _release_pipeline_resources(cap, writer, yolo_model, depth_state):
    cap.release()
    writer.release()

    del yolo_model
    del depth_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_pipeline_summary(
    video_path,
    frame_idx,
    fps,
    elapsed,
    detected_objects,
    valid_distance_objects,
    depth_state,
    runtime,
    depth_infer_every_n,
    calib_model,
    output_video_path,
    timeline_jsonl_path,
    timeline_pretty_json_path,
    live_json_path,
    birdseye_timeline_jsonl_path,
    birdseye_timeline_pretty_json_path,
    birdseye_live_json_path,
    run_summary_json_path,
    debug_timeline_jsonl_path,
    debug_timeline_pretty_json_path,
    debug_live_json_path,
    output_video_size,
):
    avg_proc_fps = frame_idx / elapsed
    distance_valid_ratio = (valid_distance_objects / detected_objects) if detected_objects else 0.0

    return {
        "run_id": RUN_ID,
        "video_path": str(video_path),
        "frames_processed": int(frame_idx),
        "source_fps": float(fps),
        "processing_seconds": float(round(elapsed, 3)),
        "avg_processing_fps": float(round(avg_proc_fps, 3)),
        "distance_valid_ratio": float(round(distance_valid_ratio, 4)),
        "depth_backend": depth_state.get("backend"),
        "runtime_device": runtime["device_str"],
        "half_precision": bool(runtime["use_half"]),
        "depth_infer_every_n": int(depth_infer_every_n),
        "selected_calibration_model": None if calib_model is None else calib_model["selected_model"],
        "outputs": {
            "annotated_video": str(output_video_path),
            "timeline_jsonl": str(timeline_jsonl_path),
            "timeline_pretty_json": None if timeline_pretty_json_path is None else str(timeline_pretty_json_path),
            "live_json": None if live_json_path is None else str(live_json_path),
            "birdseye_timeline_jsonl": None
            if birdseye_timeline_jsonl_path is None
            else str(birdseye_timeline_jsonl_path),
            "birdseye_timeline_pretty_json": None
            if birdseye_timeline_pretty_json_path is None
            else str(birdseye_timeline_pretty_json_path),
            "birdseye_live_json": None if birdseye_live_json_path is None else str(birdseye_live_json_path),
            "run_summary_json": None if run_summary_json_path is None else str(run_summary_json_path),
            "debug_timeline_jsonl": None if debug_timeline_jsonl_path is None else str(debug_timeline_jsonl_path),
            "debug_timeline_pretty_json": None
            if debug_timeline_pretty_json_path is None
            else str(debug_timeline_pretty_json_path),
            "debug_live_json": None if debug_live_json_path is None else str(debug_live_json_path),
        },
        "output_video_size": {
            "width": int(output_video_size[0]),
            "height": int(output_video_size[1]),
        },
    }


def run_mvp_pipeline(
    video_path,
    output_video_path,
    timeline_jsonl_path,
    live_json_path,
    calib_path,
    timeline_pretty_json_path=None,
    birdseye_timeline_jsonl_path=None,
    birdseye_timeline_pretty_json_path=None,
    birdseye_live_json_path=None,
    run_summary_json_path=None,
    debug_timeline_jsonl_path=None,
    debug_timeline_pretty_json_path=None,
    debug_live_json_path=None,
    max_frames=None,
):
    video_path = Path(video_path)
    output_video_path = ensure_parent(output_video_path)
    timeline_jsonl_path = ensure_parent(timeline_jsonl_path)
    if timeline_pretty_json_path is not None:
        timeline_pretty_json_path = ensure_parent(timeline_pretty_json_path)
    if live_json_path is not None:
        live_json_path = ensure_parent(live_json_path)
    if birdseye_timeline_jsonl_path is not None:
        birdseye_timeline_jsonl_path = ensure_parent(birdseye_timeline_jsonl_path)
    if birdseye_timeline_pretty_json_path is not None:
        birdseye_timeline_pretty_json_path = ensure_parent(birdseye_timeline_pretty_json_path)
    if birdseye_live_json_path is not None:
        birdseye_live_json_path = ensure_parent(birdseye_live_json_path)
    if run_summary_json_path is not None:
        run_summary_json_path = ensure_parent(run_summary_json_path)
    if debug_timeline_jsonl_path is not None:
        debug_timeline_jsonl_path = ensure_parent(debug_timeline_jsonl_path)
    if debug_timeline_pretty_json_path is not None:
        debug_timeline_pretty_json_path = ensure_parent(debug_timeline_pretty_json_path)
    if debug_live_json_path is not None:
        debug_live_json_path = ensure_parent(debug_live_json_path)
    video_io = _open_pipeline_video_io(video_path, output_video_path)
    cap = video_io["cap"]
    writer = video_io["writer"]
    fps = video_io["fps"]
    frame_w = video_io["frame_w"]
    frame_h = video_io["frame_h"]
    output_video_size = video_io["output_video_size"]
    timeline_jsonl_path.write_text("", encoding="utf-8")
    if birdseye_timeline_jsonl_path is not None:
        birdseye_timeline_jsonl_path.write_text("", encoding="utf-8")
    if debug_timeline_jsonl_path is not None:
        debug_timeline_jsonl_path.write_text("", encoding="utf-8")

    model_bundle = _load_pipeline_models(calib_path)
    runtime = model_bundle["runtime"]
    yolo_model = model_bundle["yolo_model"]
    depth_state = model_bundle["depth_state"]
    calib_model = model_bundle["calib_model"]
    depth_infer_every_n = model_bundle["depth_infer_every_n"]
    tracking_state = _create_pipeline_tracking_state()

    frame_idx = 0
    detected_objects = 0
    valid_distance_objects = 0

    start_ts = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= int(max_frames):
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        detections, next_temp_track_id, mode_used, track_exception = _detect_with_tracking(
            frame,
            yolo_model,
            tracking_state["next_temp_track_id"],
            runtime,
        )
        tracking_state["next_temp_track_id"] = next_temp_track_id
        frame_result = _process_frame_detections(
            frame=frame,
            detections=detections,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            frame_w=frame_w,
            frame_h=frame_h,
            depth_state=depth_state,
            calib_model=calib_model,
            tracking_state=tracking_state,
            depth_infer_every_n=depth_infer_every_n,
        )
        objects = frame_result["objects"]
        hook_turn_detected = frame_result["hook_turn_detected"]
        depth_target_indexes = frame_result["depth_target_indexes"]
        detected_objects += frame_result["detected_objects"]
        valid_distance_objects += frame_result["valid_distance_objects"]
        objects = _apply_detection_hold(frame, frame_idx, objects, tracking_state)

        _cleanup_stale_tracks(frame_idx, tracking_state)

        payload = build_radar_payload(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            detections=objects,
            meta={
                "run_id": RUN_ID,
                "fps_source": fps,
                "hfov_deg": HFOV_DEG,
                "frame_width": frame_w,
                "frame_height": frame_h,
            },
        )
        debug_payload = None
        if debug_timeline_jsonl_path is not None or debug_live_json_path is not None:
            debug_payload = build_radar_payload(
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                detections=objects,
                meta={
                    "run_id": RUN_ID,
                    "fps_source": fps,
                    "hfov_deg": HFOV_DEG,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                },
                include_debug=True,
            )
        _update_payload_summary(
            payload=payload,
            track_exception=track_exception,
            mode_used=mode_used,
            hook_turn_detected=hook_turn_detected,
            depth_target_indexes=depth_target_indexes,
        )
        if debug_payload is not None:
            _update_payload_summary(
                payload=debug_payload,
                track_exception=track_exception,
                mode_used=mode_used,
                hook_turn_detected=hook_turn_detected,
                depth_target_indexes=depth_target_indexes,
            )

        if live_json_path is not None:
            write_live_json(payload, live_json_path)
        _append_timeline_jsonl(payload, timeline_jsonl_path)
        if debug_live_json_path is not None and debug_payload is not None:
            write_live_json(debug_payload, debug_live_json_path)
        if debug_timeline_jsonl_path is not None and debug_payload is not None:
            _append_timeline_jsonl(debug_payload, debug_timeline_jsonl_path)
        if birdseye_live_json_path is not None:
            write_birdseye_live_json(payload, birdseye_live_json_path)
        if birdseye_timeline_jsonl_path is not None:
            _append_birdseye_timeline_jsonl(payload, birdseye_timeline_jsonl_path)
        _draw_frame_summary(frame, frame_idx, payload)

        radar_state = {
            "ema": tracking_state["radar_position_ema"],
            "history": tracking_state["radar_position_history"],
        }
        output_frame = _compose_output_frame(frame, objects, radar_state)
        output_frame = _resize_output_frame(output_frame, output_video_size)
        writer.write(output_frame)
        frame_idx += 1

    elapsed = max(time.time() - start_ts, 1e-6)
    tracking_state["last_depth_map"] = None
    _release_pipeline_resources(cap, writer, yolo_model, depth_state)

    if timeline_pretty_json_path is not None:
        write_pretty_json_from_jsonl(timeline_jsonl_path, timeline_pretty_json_path)
    if birdseye_timeline_jsonl_path is not None and birdseye_timeline_pretty_json_path is not None:
        write_pretty_json_from_jsonl(birdseye_timeline_jsonl_path, birdseye_timeline_pretty_json_path)
    if debug_timeline_jsonl_path is not None and debug_timeline_pretty_json_path is not None:
        write_pretty_json_from_jsonl(debug_timeline_jsonl_path, debug_timeline_pretty_json_path)

    summary = _build_pipeline_summary(
        video_path=video_path,
        frame_idx=frame_idx,
        fps=fps,
        elapsed=elapsed,
        detected_objects=detected_objects,
        valid_distance_objects=valid_distance_objects,
        depth_state=depth_state,
        runtime=runtime,
        depth_infer_every_n=depth_infer_every_n,
        calib_model=calib_model,
        output_video_path=output_video_path,
        timeline_jsonl_path=timeline_jsonl_path,
        timeline_pretty_json_path=timeline_pretty_json_path,
        live_json_path=live_json_path,
        birdseye_timeline_jsonl_path=birdseye_timeline_jsonl_path,
        birdseye_timeline_pretty_json_path=birdseye_timeline_pretty_json_path,
        birdseye_live_json_path=birdseye_live_json_path,
        run_summary_json_path=run_summary_json_path,
        debug_timeline_jsonl_path=debug_timeline_jsonl_path,
        debug_timeline_pretty_json_path=debug_timeline_pretty_json_path,
        debug_live_json_path=debug_live_json_path,
        output_video_size=output_video_size,
    )
    if run_summary_json_path is not None:
        write_json_file(summary, run_summary_json_path, pretty=True)
    return summary
