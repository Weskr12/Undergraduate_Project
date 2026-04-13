# -*- coding: utf-8 -*-
"""Main run_mvp_pipeline orchestration and its frame-level helpers."""

import gc
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from .config import (
    DEPTH_INFER_EVERY_N,
    DEPTH_PRO_CPU_INFER_EVERY_N,
    HFOV_DEG,
    HOOK_TURN_CLASS_ID,
)
from .depth.backends import infer_depth_map, load_depth_model
from .depth.calibration import load_calibration, raw_depth_to_distance_m
from .depth.roi import compute_object_distance
from .detection.bbox import _bearing_distance_to_xz
from .detection.yolo_tracker import _detect_with_tracking
from .output.annotations import _draw_detection, _draw_frame_summary
from .output.payload import (
    _append_timeline_jsonl,
    _build_output_object,
    _update_payload_summary,
    build_radar_payload,
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


def _estimate_detection_state(
    det_idx,
    track_id,
    bbox,
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
    distance_m = None
    approach_mps = 0.0
    ttc_s = None
    distance_band = "unknown"
    risk_level = "unknown"
    det_depth_stale = False

    if is_hook_turn:
        return {
            "raw_depth": raw_depth,
            "distance_m": distance_m,
            "approach_mps": approach_mps,
            "ttc_s": ttc_s,
            "distance_band": distance_band,
            "risk_level": "low",
            "det_depth_stale": det_depth_stale,
        }

    history = tracking_state["distance_history"][track_id] if track_id >= 0 else None
    should_estimate_distance = (det_idx in depth_target_indexes) and (tracking_state["last_depth_map"] is not None)

    if should_estimate_distance:
        raw_depth = compute_object_distance(tracking_state["last_depth_map"], bbox)
        distance_m = raw_depth_to_distance_m(raw_depth, depth_state, calib_model)

        if distance_m is not None and history is not None:
            history.append((timestamp_ms, float(distance_m)))
            distance_m = _smoothed_distance_from_history(history)
            approach_mps = _approach_from_history(history, distance_m, timestamp_ms)

        det_depth_stale = bool(frame_depth_stale)
    else:
        distance_m = _smoothed_distance_from_history(history)
        approach_mps = _approach_from_history(history, distance_m, timestamp_ms)
        det_depth_stale = bool(frame_depth_stale or history or (det_idx not in depth_target_indexes))

    distance_band = _distance_band(distance_m)
    if stable_for_ttc:
        ttc_s = _compute_ttc_s(distance_m, approach_mps)
        risk_level = _risk_level_from_ttc(distance_m, approach_mps, ttc_s)
    else:
        risk_level = _risk_level(distance_m, approach_mps)

    return {
        "raw_depth": raw_depth,
        "distance_m": distance_m,
        "approach_mps": approach_mps,
        "ttc_s": ttc_s,
        "distance_band": distance_band,
        "risk_level": risk_level,
        "det_depth_stale": det_depth_stale,
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

        pos_x, pos_z = _bearing_distance_to_xz(detection_state["distance_m"], bearing_deg)
        velocity_xz = (None, None)
        if track_id >= 0 and pos_x is not None and pos_z is not None:
            pos_history = tracking_state["position_xz_history"][track_id]
            pos_history.append((timestamp_ms, float(pos_x), float(pos_z)))
            velocity_xz = _velocity_from_position_history(pos_history)

        if track_id >= 0:
            tracking_state["last_seen_frame"][track_id] = frame_idx

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
            position_xz=(pos_x, pos_z),
            velocity_xz=velocity_xz,
            track_status=track_status,
        )
        _smooth_detection_display(obj, tracking_state["display_track_state"], frame_w, frame_h)
        objects.append(obj)
        _draw_detection(frame, obj)

    return {
        "objects": objects,
        "hook_turn_detected": hook_turn_detected,
        "depth_target_indexes": depth_target_indexes,
        "detected_objects": detected_objects,
        "valid_distance_objects": valid_distance_objects,
    }


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
    live_json_path,
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
            "live_json": str(live_json_path),
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
    max_frames=None,
):
    video_path = Path(video_path)
    output_video_path = ensure_parent(output_video_path)
    timeline_jsonl_path = ensure_parent(timeline_jsonl_path)
    live_json_path = ensure_parent(live_json_path)
    video_io = _open_pipeline_video_io(video_path, output_video_path)
    cap = video_io["cap"]
    writer = video_io["writer"]
    fps = video_io["fps"]
    frame_w = video_io["frame_w"]
    frame_h = video_io["frame_h"]
    output_video_size = video_io["output_video_size"]
    timeline_jsonl_path.write_text("", encoding="utf-8")

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
        _update_payload_summary(
            payload=payload,
            track_exception=track_exception,
            mode_used=mode_used,
            hook_turn_detected=hook_turn_detected,
            depth_target_indexes=depth_target_indexes,
        )

        write_live_json(payload, live_json_path)
        _append_timeline_jsonl(payload, timeline_jsonl_path)
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

    return _build_pipeline_summary(
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
        live_json_path=live_json_path,
        output_video_size=output_video_size,
    )
