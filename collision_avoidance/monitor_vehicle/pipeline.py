# -*- coding: utf-8 -*-
"""Main run_mvp_pipeline orchestration and its frame-level helpers."""

import gc
import math
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from .config import (
    ACTIVE_DUPLICATE_SUPPRESS_CENTER_RATIO,
    ACTIVE_DUPLICATE_SUPPRESS_CONTAINMENT_THRESHOLD,
    ACTIVE_DUPLICATE_SUPPRESS_IOU_THRESHOLD,
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
    HELD_ID_REUSE_INTERSECTION_RATIO_THRESHOLD,
    HELD_ID_REUSE_IOU_THRESHOLD,
    HELD_OVERLAP_NORMALIZATION_MODE,
    HELD_SUPPRESS_INTERSECTION_RATIO_THRESHOLD,
    HELD_SUPPRESS_IOU_THRESHOLD,
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
    _smooth_display_distance,
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
        "display_distance_m": None,
        "display_depth_used_m": None,
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


def _bbox_iou_xyxy(a, b):
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def _bbox_overlap_metrics_xyxy(a, b):
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0, 0.0

    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    iou = 0.0 if union <= 0.0 else float(inter_area / union)
    normalization_mode = str(HELD_OVERLAP_NORMALIZATION_MODE).strip().lower()
    if normalization_mode in {"held_area", "a_area", "area_a"}:
        normalized_area = area_a
    elif normalization_mode in {"active_area", "b_area", "area_b"}:
        normalized_area = area_b
    elif normalization_mode in {"max_area", "max"}:
        normalized_area = max(area_a, area_b)
    elif normalization_mode in {"union", "iou"}:
        normalized_area = union
    else:
        normalized_area = min(area_a, area_b)
    intersection_ratio = 0.0 if normalized_area <= 0.0 else float(inter_area / normalized_area)
    return iou, intersection_ratio


def _bbox_area_xyxy(bbox):
    if bbox is None or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center_xyxy(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_containment_ratio_xyxy(a, b):
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    min_area = min(_bbox_area_xyxy(a), _bbox_area_xyxy(b))
    return 0.0 if min_area <= 0.0 else float(inter_area / min_area)


def _bbox_center_distance_ratio_xyxy(a, b):
    center_a = _bbox_center_xyxy(a)
    center_b = _bbox_center_xyxy(b)
    if center_a is None or center_b is None:
        return 1.0

    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ref_w = max(1.0, min(max(0.0, ax2 - ax1), max(0.0, bx2 - bx1)))
    ref_h = max(1.0, min(max(0.0, ay2 - ay1), max(0.0, by2 - by1)))
    dx = abs(center_a[0] - center_b[0]) / ref_w
    dy = abs(center_a[1] - center_b[1]) / ref_h
    return float(math.hypot(dx, dy))


def _same_output_class(a, b):
    return (
        int(a.get("class_id", -1)) == int(b.get("class_id", -2))
        or str(a.get("class_name", "")).lower() == str(b.get("class_name", "")).lower()
    )


def _is_suppressed_by_active_overlap(
    held_obj,
    active_objects,
    iou_threshold=HELD_SUPPRESS_IOU_THRESHOLD,
    intersection_ratio_threshold=HELD_SUPPRESS_INTERSECTION_RATIO_THRESHOLD,
):
    if iou_threshold <= 0 and intersection_ratio_threshold <= 0:
        return False

    held_bbox = held_obj.get("bbox_xyxy")
    for obj in active_objects:
        if not _same_output_class(held_obj, obj):
            continue
        iou, intersection_ratio = _bbox_overlap_metrics_xyxy(held_bbox, obj.get("bbox_xyxy"))
        if iou >= float(iou_threshold) or intersection_ratio >= float(intersection_ratio_threshold):
            return True
    return False


def _is_same_frame_duplicate_detection(a, b):
    if int(a.get("class_id", -1)) != int(b.get("class_id", -2)):
        return False
    if int(a.get("class_id", -1)) == HOOK_TURN_CLASS_ID:
        return False

    bbox_a = a.get("bbox_xyxy")
    bbox_b = b.get("bbox_xyxy")
    iou, _intersection_ratio = _bbox_overlap_metrics_xyxy(bbox_a, bbox_b)
    if iou >= float(ACTIVE_DUPLICATE_SUPPRESS_IOU_THRESHOLD):
        return True

    containment = _bbox_containment_ratio_xyxy(bbox_a, bbox_b)
    if containment >= float(ACTIVE_DUPLICATE_SUPPRESS_CONTAINMENT_THRESHOLD):
        return True

    center_ratio = _bbox_center_distance_ratio_xyxy(bbox_a, bbox_b)
    return containment >= 0.6 and center_ratio <= float(ACTIVE_DUPLICATE_SUPPRESS_CENTER_RATIO)


def _active_detection_keep_score(det, tracking_state):
    track_id = int(det.get("track_id", -1))
    hit_count = int(tracking_state["track_hit_count"].get(track_id, 0)) if track_id >= 0 else 0
    confidence = max(0.0, float(det.get("confidence", 0.0)))
    area = _bbox_area_xyxy(det.get("bbox_xyxy"))
    # Prefer a stable existing ID, then detector confidence, then the fuller box.
    return (hit_count, confidence, area)


def _suppress_same_frame_duplicate_detections(detections, tracking_state):
    if (
        float(ACTIVE_DUPLICATE_SUPPRESS_IOU_THRESHOLD) <= 0
        and float(ACTIVE_DUPLICATE_SUPPRESS_CONTAINMENT_THRESHOLD) <= 0
    ):
        return detections

    kept = []
    for det in detections:
        replace_idx = None
        drop_current = False
        for idx, kept_det in enumerate(kept):
            if not _is_same_frame_duplicate_detection(det, kept_det):
                continue

            current_score = _active_detection_keep_score(det, tracking_state)
            kept_score = _active_detection_keep_score(kept_det, tracking_state)
            if current_score > kept_score:
                replace_idx = idx
            else:
                drop_current = True
            break

        if replace_idx is not None:
            kept[replace_idx] = det
        elif not drop_current:
            kept.append(det)

    return kept


def _output_object_keep_score(obj):
    hit_count = int(obj.get("hit_count", 0))
    confidence = max(0.0, float(obj.get("track_confidence", obj.get("confidence", 0.0)) or 0.0))
    area = _bbox_area_xyxy(obj.get("bbox_xyxy"))
    return (hit_count, confidence, area)


def _suppress_same_frame_duplicate_objects(objects):
    kept = []
    dropped_track_ids = []
    for obj in objects:
        replace_idx = None
        drop_current = False
        for idx, kept_obj in enumerate(kept):
            if not _same_output_class(obj, kept_obj):
                continue
            if int(obj.get("class_id", -1)) == HOOK_TURN_CLASS_ID:
                continue
            if not _is_same_frame_duplicate_detection(obj, kept_obj):
                continue

            current_score = _output_object_keep_score(obj)
            kept_score = _output_object_keep_score(kept_obj)
            if current_score > kept_score:
                dropped_track_ids.append(int(kept_obj.get("track_id", -1)))
                replace_idx = idx
            else:
                dropped_track_ids.append(int(obj.get("track_id", -1)))
                drop_current = True
            break

        if replace_idx is not None:
            kept[replace_idx] = obj
        elif not drop_current:
            kept.append(obj)

    return kept, [track_id for track_id in dropped_track_ids if track_id >= 0]


def _is_vehicle_track_entry(entry):
    return int(entry.get("class_id", -1)) != HOOK_TURN_CLASS_ID and not bool(entry.get("hook_turn_detected"))


def _find_held_overlap_reuse_track_id(det, frame_idx, tracking_state):
    track_id = int(det.get("track_id", -1))
    if track_id < 0 or track_id in tracking_state["last_seen_frame"]:
        return None
    if int(det.get("class_id", -1)) == HOOK_TURN_CLASS_ID:
        return None

    bbox = det.get("bbox_xyxy")
    best_source_track_id = None
    best_score = -1.0
    best_last_seen_frame = -1

    for source_track_id, last_seen_frame in list(tracking_state["last_seen_frame"].items()):
        missing_frames = int(frame_idx) - int(last_seen_frame)
        if missing_frames <= 0 or missing_frames > DETECTION_HOLD_FRAMES:
            continue

        previous_obj = tracking_state["last_output_objects"].get(source_track_id)
        if previous_obj is None or not _is_vehicle_track_entry(previous_obj):
            continue

        iou, intersection_ratio = _bbox_overlap_metrics_xyxy(bbox, previous_obj.get("bbox_xyxy"))
        if (
            iou < float(HELD_ID_REUSE_IOU_THRESHOLD)
            and intersection_ratio < float(HELD_ID_REUSE_INTERSECTION_RATIO_THRESHOLD)
        ):
            continue

        score = max(iou, intersection_ratio)
        if score > best_score or (score == best_score and int(last_seen_frame) > best_last_seen_frame):
            best_score = score
            best_last_seen_frame = int(last_seen_frame)
            best_source_track_id = int(source_track_id)

    return best_source_track_id


def _drop_track_transient_state(track_id, tracking_state):
    if track_id < 0:
        return

    for key in (
        "distance_history",
        "distance_confidence_history",
        "rejected_depth_candidates",
        "distance_band_state",
        "track_observation_history",
        "display_track_state",
        "display_distance_state",
        "class_vote_history",
        "class_vote_state",
        "track_hit_count",
        "last_seen_frame",
        "last_output_objects",
        "radar_position_ema",
        "radar_position_history",
        "position_xz_history",
    ):
        tracking_state[key].pop(track_id, None)


def _reuse_held_track_id_for_new_overlap(det, frame_idx, tracking_state):
    reuse_track_id = _find_held_overlap_reuse_track_id(det, frame_idx, tracking_state)
    if reuse_track_id is None:
        return det

    raw_track_id = int(det.get("track_id", -1))
    stabilized_det = dict(det)
    stabilized_det["raw_track_id"] = raw_track_id
    stabilized_det["track_id"] = int(reuse_track_id)
    stabilized_det["track_id_source"] = "held_overlap_reuse"
    _drop_track_transient_state(raw_track_id, tracking_state)
    return stabilized_det


def _class_vote_key_from_entry(entry):
    if len(entry) >= 2:
        return int(entry[0]), str(entry[1])
    raise ValueError(f"Invalid class vote entry: {entry!r}")


def _class_vote_confidence_from_entry(entry):
    if len(entry) >= 3:
        return float(entry[2])
    return 1.0


def _inherit_class_vote_from_held_overlap(det, frame_idx, tracking_state):
    track_id = int(det.get("track_id", -1))
    if track_id < 0 or tracking_state["class_vote_history"].get(track_id):
        return

    bbox = det.get("bbox_xyxy")
    best_source_track_id = None
    best_iou = float(HELD_SUPPRESS_IOU_THRESHOLD)

    for source_track_id, last_seen_frame in list(tracking_state["last_seen_frame"].items()):
        if int(source_track_id) == track_id:
            continue

        missing_frames = int(frame_idx) - int(last_seen_frame)
        if missing_frames <= 0 or missing_frames > DETECTION_HOLD_FRAMES:
            continue

        previous_obj = tracking_state["last_output_objects"].get(source_track_id)
        if previous_obj is None or previous_obj.get("hook_turn_detected"):
            continue

        overlap = _bbox_iou_xyxy(bbox, previous_obj.get("bbox_xyxy"))
        if overlap >= best_iou:
            best_iou = overlap
            best_source_track_id = source_track_id

    if best_source_track_id is None:
        return

    source_history = tracking_state["class_vote_history"].get(best_source_track_id)
    if source_history:
        tracking_state["class_vote_history"][track_id].extend(source_history)

    source_state = tracking_state["class_vote_state"].get(best_source_track_id)
    if source_state is not None:
        tracking_state["class_vote_state"][track_id] = source_state


def _apply_track_class_vote(det, tracking_state):
    track_id = int(det.get("track_id", -1))
    if track_id < 0:
        return det

    class_key = (int(det["class_id"]), str(det["class_name"]))
    confidence = max(0.0, float(det.get("confidence", 0.0)))
    history = tracking_state["class_vote_history"][track_id]
    history.append((class_key[0], class_key[1], confidence))

    scores = defaultdict(float)
    for entry in history:
        scores[_class_vote_key_from_entry(entry)] += _class_vote_confidence_from_entry(entry)

    max_score = max(scores.values())
    winners = [key for key, score in scores.items() if score == max_score]
    previous_winner = tracking_state["class_vote_state"].get(track_id)
    if previous_winner in winners:
        voted_class_id, voted_class_name = previous_winner
    elif class_key in winners:
        voted_class_id, voted_class_name = class_key
    else:
        voted_class_id, voted_class_name = sorted(winners, key=lambda item: (item[0], item[1]))[0]

    tracking_state["class_vote_state"][track_id] = (voted_class_id, voted_class_name)
    if voted_class_id == class_key[0] and voted_class_name == class_key[1]:
        return det

    voted_det = dict(det)
    voted_det["class_id"] = int(voted_class_id)
    voted_det["class_name"] = str(voted_class_name)
    return voted_det


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
        and height <= YOLO_FAR_UNRELIABLE_MAX_BBOX_HEIGHT_PX
        and area <= YOLO_FAR_UNRELIABLE_MAX_BBOX_AREA_PX
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

    display_distance_m = _smooth_display_distance(
        track_id,
        {**depth_debug, "distance_m": distance_m},
        tracking_state["display_distance_state"],
    )
    display_band = _distance_band(display_distance_m)
    display_depth_used_m = _depth_used_for_position(display_distance_m, display_band)
    depth_debug["display_distance_m"] = display_distance_m
    depth_debug["display_depth_used_m"] = display_depth_used_m

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
    detections = _suppress_same_frame_duplicate_detections(detections, tracking_state)
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
        det = _reuse_held_track_id_for_new_overlap(det, frame_idx, tracking_state)
        _inherit_class_vote_from_held_overlap(det, frame_idx, tracking_state)
        det = _apply_track_class_vote(det, tracking_state)
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
        if "raw_track_id" in det:
            obj["raw_track_id"] = int(det["raw_track_id"])
            obj["track_id_source"] = str(det.get("track_id_source", "unknown"))
        obj["hit_count"] = hit_count
        obj["track_state"] = track_state
        obj["missed_frames"] = 0
        _smooth_detection_display(obj, tracking_state["display_track_state"], frame_w, frame_h)
        position_depth_m = obj.get("display_depth_used_m")
        if position_depth_m is None:
            position_depth_m = obj.get("depth_used_m")
        pos_x, pos_z = _bearing_distance_to_xz(position_depth_m, obj.get("bearing_deg"))
        velocity_xz = (None, None)
        if track_id >= 0 and pos_x is not None and pos_z is not None:
            pos_history = tracking_state["position_xz_history"][track_id]
            pos_history.append((timestamp_ms, float(pos_x), float(pos_z)))
            velocity_xz = _velocity_from_position_history(pos_history)
        obj["position_m"] = _position_dict_from_xz(pos_x, pos_z)
        obj["velocity_mps"] = _velocity_dict_from_xz(*velocity_xz)
        objects.append(obj)

    objects, dropped_track_ids = _suppress_same_frame_duplicate_objects(objects)
    for dropped_track_id in dropped_track_ids:
        _drop_track_transient_state(dropped_track_id, tracking_state)
    for obj in objects:
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
        held_obj["display_distance_m"] = held_obj.get("display_distance_m", held_obj.get("distance_m"))
        held_obj["display_depth_used_m"] = held_obj.get("display_depth_used_m", held_obj.get("depth_used_m"))
        held_obj["approach_mps"] = 0.0
        held_obj["ttc_s"] = None
        if _is_suppressed_by_active_overlap(held_obj, objects):
            continue
        held_objects.append(held_obj)

    if held_objects:
        combined_objects, dropped_track_ids = _suppress_same_frame_duplicate_objects(objects + held_objects)
        for dropped_track_id in dropped_track_ids:
            _drop_track_transient_state(dropped_track_id, tracking_state)
        if DRAW_HELD_DETECTIONS:
            active_track_ids = {int(obj.get("track_id", -1)) for obj in objects}
            for obj in combined_objects:
                if int(obj.get("track_id", -1)) not in active_track_ids:
                    _draw_detection(frame, obj)
        return combined_objects

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
    start_frame=None,
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

    start_frame_idx = 0 if start_frame is None else max(0, int(start_frame))
    if start_frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    frame_idx = start_frame_idx
    frames_processed = 0
    detected_objects = 0
    valid_distance_objects = 0

    start_ts = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frames_processed >= int(max_frames):
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
        frames_processed += 1

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
        frame_idx=frames_processed,
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
