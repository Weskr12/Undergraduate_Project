# -*- coding: utf-8 -*-
"""Build per-frame JSON payloads, write live/timeline files, validate schema."""

import json

from ..utils import ensure_parent


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
        "confidence": round(float(det["confidence"]), 4),
        "bbox_xyxy": bbox,
        "bearing_deg": round(float(bearing_deg), 2),
        "distance_m": None
        if detection_state["distance_m"] is None
        else round(float(detection_state["distance_m"]), 3),
        "distance_band": detection_state["distance_band"],
        "approach_mps": round(float(detection_state["approach_mps"]), 3),
        "ttc_s": None if detection_state["ttc_s"] is None else round(float(detection_state["ttc_s"]), 3),
        "stable_for_ttc": bool(stable_for_ttc),
        "risk_level": detection_state["risk_level"],
        "hook_turn_detected": bool(is_hook_turn),
        "depth_stale": bool(detection_state["det_depth_stale"]),
        "position_m": position_m,
        "velocity_mps": velocity_mps,
        "track_status": str(track_status),
    }


def build_radar_payload(frame_idx, timestamp_ms, detections, meta):
    valid_distances = [d["distance_m"] for d in detections if d.get("distance_m") is not None]
    valid_ttc = [d["ttc_s"] for d in detections if d.get("ttc_s") is not None]
    stable_ttc_target_count = sum(1 for d in detections if d.get("stable_for_ttc", False))
    high_risk_count = sum(1 for d in detections if d.get("risk_level") in {"high", "critical"})
    hook_turn_detected = any(d.get("hook_turn_detected", False) for d in detections)

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
        "objects": detections,
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


def _update_payload_summary(payload, track_exception, mode_used, hook_turn_detected, depth_target_indexes):
    if track_exception is not None:
        payload["summary"]["track_warning"] = str(track_exception)
    payload["summary"]["mode_used"] = mode_used
    payload["summary"]["hook_turn_detected"] = bool(hook_turn_detected)
    payload["summary"]["depth_target_count"] = int(len(depth_target_indexes))
    payload["summary"]["depth_skipped"] = bool(not depth_target_indexes)


def write_live_json(payload, live_json_path):
    path = ensure_parent(live_json_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _append_timeline_jsonl(payload, timeline_jsonl_path):
    path = ensure_parent(timeline_jsonl_path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
