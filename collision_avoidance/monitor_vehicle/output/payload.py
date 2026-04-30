# -*- coding: utf-8 -*-
"""Build per-frame JSON payloads, write live/timeline files, validate schema."""

import json

from ..utils import ensure_parent


LEGACY_OBJECT_KEYS = (
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "bbox_xyxy",
    "bearing_deg",
    "distance_m",
    "distance_band",
    "approach_mps",
    "ttc_s",
    "stable_for_ttc",
    "risk_level",
    "hook_turn_detected",
    "depth_stale",
    "depth_raw_m",
    "depth_smooth_m",
    "depth_used_m",
    "depth_source",
    "depth_quality",
    "depth_valid_ratio",
    "depth_roi",
    "depth_sample_method",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px",
    "depth_reliable",
    "unity_z_raw",
    "unity_z_used",
    "is_outlier",
    "outlier_reason",
    "missing_frames",
    "position_m",
    "velocity_mps",
    "track_status",
)


def _round_or_none(value, digits=3):
    if value is None:
        return None
    return round(float(value), int(digits))


def _build_output_object(
    det,
    bbox,
    bearing_deg,
    stable_for_ttc,
    detection_state,
    is_hook_turn,
    position_xz=(None, None),
    velocity_xz=(None, None),
    track_status="active",
):
    pos_x, pos_z = position_xz
    vel_x, vel_z = velocity_xz
    confidence_scale = float(detection_state.get("confidence_scale", 1.0))
    position_m = None if pos_x is None or pos_z is None else {
        "x": round(float(pos_x), 3),
        "z": round(float(pos_z), 3),
    }
    velocity_mps = None if vel_x is None or vel_z is None else {
        "vx": round(float(vel_x), 3),
        "vz": round(float(vel_z), 3),
    }
    return {
        "track_id": int(det["track_id"]),
        "class_id": int(det["class_id"]),
        "class_name": str(det["class_name"]),
        "confidence": round(float(det["confidence"]) * confidence_scale, 4),
        "bbox_xyxy": bbox,
        "bearing_deg": round(float(bearing_deg), 2),
        "distance_m": _round_or_none(detection_state["distance_m"], 3),
        "distance_band": detection_state["distance_band"],
        "approach_mps": round(float(detection_state["approach_mps"]), 3),
        "ttc_s": None if detection_state["ttc_s"] is None else round(float(detection_state["ttc_s"]), 3),
        "stable_for_ttc": bool(stable_for_ttc),
        "risk_level": detection_state["risk_level"],
        "hook_turn_detected": bool(is_hook_turn),
        "depth_stale": bool(detection_state["det_depth_stale"]),
        "depth_raw_m": _round_or_none(detection_state.get("depth_raw_m"), 3),
        "depth_smooth_m": _round_or_none(detection_state.get("depth_smooth_m"), 3),
        "depth_used_m": _round_or_none(detection_state.get("depth_used_m"), 3),
        "raw_distance_m": _round_or_none(detection_state.get("raw_distance_m"), 3),
        "distance_after_gate_m": _round_or_none(detection_state.get("distance_after_gate_m"), 3),
        "distance_smooth_m": _round_or_none(detection_state.get("distance_smooth_m"), 3),
        "depth_source": str(detection_state.get("depth_source", "unknown")),
        "depth_quality": str(detection_state.get("depth_quality", "bad")),
        "depth_valid_ratio": _round_or_none(detection_state.get("depth_valid_ratio"), 4),
        "depth_roi": detection_state.get("depth_roi"),
        "roi_xyxy": detection_state.get("roi_xyxy"),
        "roi_pixel_count": int(detection_state.get("roi_pixel_count", 0) or 0),
        "depth_iqr": _round_or_none(detection_state.get("depth_iqr"), 3),
        "depth_percentile_used": _round_or_none(detection_state.get("depth_percentile_used"), 1),
        "depth_p25": _round_or_none(detection_state.get("depth_p25"), 3),
        "depth_p35": _round_or_none(detection_state.get("depth_p35"), 3),
        "depth_p40": _round_or_none(detection_state.get("depth_p40"), 3),
        "depth_p50": _round_or_none(detection_state.get("depth_p50"), 3),
        "depth_sample_method": detection_state.get("depth_sample_method"),
        "bbox_width_px": detection_state.get("bbox_width_px"),
        "bbox_height_px": detection_state.get("bbox_height_px"),
        "bbox_area_px": detection_state.get("bbox_area_px"),
        "bbox_area": detection_state.get("bbox_area"),
        "depth_reliable": bool(detection_state.get("depth_reliable", False)),
        "depth_confidence": _round_or_none(detection_state.get("depth_confidence"), 4),
        "unity_z_raw": _round_or_none(detection_state.get("unity_z_raw"), 3),
        "unity_z_used": _round_or_none(detection_state.get("unity_z_used"), 3),
        "is_outlier": bool(detection_state.get("is_outlier", False)),
        "outlier_reason": detection_state.get("outlier_reason"),
        "gate_decision": str(detection_state.get("gate_decision", "missing")),
        "gate_reason": detection_state.get("gate_reason"),
        "missing_frames": int(detection_state.get("missing_frames", 0)),
        "missed_frames": int(detection_state.get("missed_frames", detection_state.get("missing_frames", 0)) or 0),
        "distance_source": str(detection_state.get("distance_source", "missing")),
        "position_m": position_m,
        "velocity_mps": velocity_mps,
        "track_status": str(track_status),
    }


def _legacy_output_object(det):
    return {key: det.get(key) for key in LEGACY_OBJECT_KEYS if key in det}


def _debug_output_object(det, frame_idx, timestamp_ms):
    debug_obj = dict(det)
    debug_obj["frame_idx"] = int(frame_idx)
    debug_obj["timestamp_ms"] = int(timestamp_ms)
    debug_obj["vehicle_type"] = str(det.get("class_name", "unknown"))
    return debug_obj


def build_radar_payload(frame_idx, timestamp_ms, detections, meta, include_debug=False):
    output_detections = [
        _debug_output_object(det, frame_idx, timestamp_ms) if include_debug else _legacy_output_object(det)
        for det in detections
    ]

    valid_distances = [d["distance_m"] for d in output_detections if d.get("distance_m") is not None]
    valid_ttc = [d["ttc_s"] for d in output_detections if d.get("ttc_s") is not None]
    stable_ttc_target_count = sum(1 for d in output_detections if d.get("stable_for_ttc", False))
    high_risk_count = sum(1 for d in output_detections if d.get("risk_level") in {"high", "critical"})
    hook_turn_detected = any(d.get("hook_turn_detected", False) for d in output_detections)

    payload = {
        "run_id": meta["run_id"],
        "frame_idx": int(frame_idx),
        "timestamp_ms": int(timestamp_ms),
        "fps_source": float(meta["fps_source"]),
        "camera": {
            "hfov_deg": float(meta["hfov_deg"]),
            "frame_width": int(meta["frame_width"]),
            "frame_height": int(meta["frame_height"]),
            "coord_system": "ego_local_unity",
        },
        "objects": output_detections,
        "summary": {
            "object_count": int(len(detections)),
            "min_distance_m": float(min(valid_distances)) if valid_distances else None,
            "min_ttc_s": float(min(valid_ttc)) if valid_ttc else None,
            "stable_ttc_target_count": int(stable_ttc_target_count),
            "high_risk_count": int(high_risk_count),
            "hook_turn_detected": bool(hook_turn_detected),
        },
    }
    return payload


def _build_birdseye_object(obj):
    position_m = obj.get("position_m")
    x_m = None
    z_m = None
    if isinstance(position_m, dict):
        x_m = position_m.get("x")
        z_m = position_m.get("z")

    return {
        "id": int(obj["track_id"]),
        "vehicle_type": str(obj["class_name"]),
        "position_m": None if x_m is None or z_m is None else {
            "x": round(float(x_m), 3),
            "z": round(float(z_m), 3),
        },
        "distance_m": obj.get("distance_m"),
        "bearing_deg": obj.get("bearing_deg"),
        "risk_level": obj.get("risk_level"),
    }


def build_birdseye_payload(payload):
    objects = [
        _build_birdseye_object(obj)
        for obj in payload.get("objects", [])
        if not obj.get("hook_turn_detected", False)
    ]
    valid_distances = [obj["distance_m"] for obj in objects if obj.get("distance_m") is not None]
    high_risk_count = sum(1 for obj in objects if obj.get("risk_level") in {"high", "critical"})

    return {
        "run_id": payload["run_id"],
        "frame_idx": int(payload["frame_idx"]),
        "timestamp_ms": int(payload["timestamp_ms"]),
        "fps_source": float(payload["fps_source"]),
        "camera": payload["camera"],
        "objects": objects,
        "summary": {
            "object_count": int(len(objects)),
            "min_distance_m": float(min(valid_distances)) if valid_distances else None,
            "high_risk_count": int(high_risk_count),
        },
    }


def _update_payload_summary(payload, track_exception, mode_used, hook_turn_detected, depth_target_indexes):
    if track_exception is not None:
        payload["summary"]["track_warning"] = str(track_exception)
    payload["summary"]["mode_used"] = mode_used
    payload["summary"]["hook_turn_detected"] = bool(hook_turn_detected)
    payload["summary"]["depth_target_count"] = int(len(depth_target_indexes))
    payload["summary"]["depth_skipped"] = bool(not depth_target_indexes)


def write_json_file(payload, json_path, pretty=True, indent=2):
    path = ensure_parent(json_path)
    dump_kwargs = {"ensure_ascii": False}
    if pretty:
        dump_kwargs["indent"] = int(indent)
    else:
        dump_kwargs["separators"] = (",", ":")
    path.write_text(json.dumps(payload, **dump_kwargs), encoding="utf-8")


def write_live_json(payload, live_json_path, pretty=True, indent=2):
    write_json_file(payload, live_json_path, pretty=pretty, indent=indent)


def _append_timeline_jsonl(payload, timeline_jsonl_path):
    path = ensure_parent(timeline_jsonl_path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_birdseye_live_json(payload, live_json_path, indent=2):
    birdseye_payload = build_birdseye_payload(payload)
    write_live_json(birdseye_payload, live_json_path, pretty=True, indent=indent)


def _append_birdseye_timeline_jsonl(payload, timeline_jsonl_path):
    birdseye_payload = build_birdseye_payload(payload)
    _append_timeline_jsonl(birdseye_payload, timeline_jsonl_path)


def write_pretty_json_from_jsonl(timeline_jsonl_path, pretty_json_path, indent=2):
    jsonl_path = ensure_parent(timeline_jsonl_path)
    pretty_path = ensure_parent(pretty_json_path)
    records = []

    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    write_json_file(records, pretty_path, pretty=True, indent=indent)


def _validate_payload_schema(payload):
    top_keys = {"run_id", "frame_idx", "timestamp_ms", "fps_source", "camera", "objects", "summary"}
    camera_keys = {"hfov_deg", "frame_width", "frame_height", "coord_system"}
    object_keys = {
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "bbox_xyxy",
        "bearing_deg",
        "distance_m",
        "distance_band",
        "approach_mps",
        "ttc_s",
        "stable_for_ttc",
        "risk_level",
        "hook_turn_detected",
        "depth_stale",
        "position_m",
        "velocity_mps",
        "track_status",
    }

    assert top_keys.issubset(payload.keys()), f"Missing top-level keys: {top_keys - set(payload.keys())}"
    assert camera_keys.issubset(payload["camera"].keys()), (
        f"Missing camera keys: {camera_keys - set(payload['camera'].keys())}"
    )
    for obj in payload["objects"]:
        assert object_keys.issubset(obj.keys()), f"Missing object keys: {object_keys - set(obj.keys())}"
