# -*- coding: utf-8 -*-
"""Replay depth gating with ROI percentile alternatives from debug JSONL."""

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_PARENT = ROOT / "collision_avoidance"
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from monitor_vehicle.config import (  # noqa: E402
    DEPTH_MAX_ABS_JUMP_M,
    DEPTH_MAX_PLAUSIBLE_SPEED_MPS,
    DEPTH_MAX_REL_JUMP,
    DEPTH_MIN_CONFIDENCE_FOR_BASELINE,
    DEPTH_RECOVERY_MAX_CANDIDATES,
    DEPTH_RECOVERY_MIN_CONFIDENCE,
    DEPTH_RECOVERY_MIN_CONSECUTIVE,
    DEPTH_RELIABLE_MAX_M,
    DISTANCE_HISTORY_LEN,
    MIN_DEPTH,
    MIN_RELIABLE_BBOX_AREA_PX,
    SMOOTH_WINDOW,
)


PERCENTILES = (25, 35, 40, 50)


def _median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _smooth(history):
    if not history:
        return None
    return _median([value for _, value in list(history)[-SMOOTH_WINDOW:]])


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _should_accept(raw_m, previous_smooth, history, timestamp_ms, depth_quality):
    reasons = []
    if depth_quality == "bad":
        reasons.append("roi_quality_bad")
    if not _finite(raw_m):
        reasons.append("invalid_depth")
    elif raw_m <= MIN_DEPTH:
        reasons.append("non_positive_depth")

    if previous_smooth is not None and _finite(raw_m):
        jump_m = abs(float(raw_m) - float(previous_smooth))
        if jump_m > DEPTH_MAX_ABS_JUMP_M:
            reasons.append("abs_jump")
        if previous_smooth > MIN_DEPTH and jump_m / max(float(previous_smooth), MIN_DEPTH) > DEPTH_MAX_REL_JUMP:
            reasons.append("rel_jump")
        if history:
            previous_ts = int(list(history)[-1][0])
            dt_s = max((int(timestamp_ms) - previous_ts) / 1000.0, 1e-6)
            if jump_m / dt_s > DEPTH_MAX_PLAUSIBLE_SPEED_MPS:
                reasons.append("implied_speed")
    return reasons


def _recovery_ok(candidates, previous_smooth):
    if previous_smooth is None or len(candidates) < DEPTH_RECOVERY_MIN_CONSECUTIVE:
        return False
    recent = list(candidates)[-DEPTH_RECOVERY_MIN_CONSECUTIVE:]
    raw_values = [item["raw"] for item in recent]
    if not all(_finite(value) for value in raw_values):
        return False
    if not all(item["conf"] >= DEPTH_RECOVERY_MIN_CONFIDENCE for item in recent):
        return False
    if not all(item["area"] >= MIN_RELIABLE_BBOX_AREA_PX for item in recent):
        return False
    if not all(value < previous_smooth for value in raw_values):
        return False
    return all(raw_values[idx] <= raw_values[idx - 1] + 1.0 for idx in range(1, len(raw_values)))


def replay_percentile(records, percentile):
    key = f"depth_p{percentile}"
    has_percentile_column = any(
        key in obj
        for payload in records
        for obj in payload.get("objects", [])
    )
    if percentile != 35 and not has_percentile_column:
        return {
            "percentile": f"p{percentile}",
            "valid_distance_frame_ratio": "n/a",
            "distance_na_count": "n/a",
            "rejected_by_gate_count": "n/a",
            "closing_tracks_down": "n/a",
            "max_single_frame_jump_m": "n/a",
            "avg_depth_confidence": "n/a",
            "track_interruptions": "n/a",
        }

    histories = defaultdict(lambda: deque(maxlen=DISTANCE_HISTORY_LEN))
    rejected_candidates = defaultdict(lambda: deque(maxlen=DEPTH_RECOVERY_MAX_CANDIDATES))
    previous_frame_by_track = {}
    smooth_by_track = defaultdict(list)
    stats = {
        "rows": 0,
        "valid": 0,
        "na": 0,
        "rejected": 0,
        "max_jump": 0.0,
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "track_interruptions": 0,
    }

    for payload in records:
        frame_idx = int(payload["frame_idx"])
        timestamp_ms = int(payload["timestamp_ms"])
        for obj in payload.get("objects", []):
            if obj.get("hook_turn_detected") or int(obj.get("track_id", -1)) < 0:
                continue

            track_id = int(obj["track_id"])
            if track_id in previous_frame_by_track and frame_idx - previous_frame_by_track[track_id] > 1:
                stats["track_interruptions"] += 1
            previous_frame_by_track[track_id] = frame_idx

            raw_m = obj.get(key)
            if raw_m is None and percentile == 35:
                raw_m = obj.get("raw_distance_m", obj.get("depth_raw_m"))
            depth_quality = obj.get("depth_quality")
            depth_confidence = obj.get("depth_confidence")
            if depth_confidence is None:
                depth_confidence = 0.0 if depth_quality == "bad" else 0.6 if depth_quality == "weak" else 1.0
            bbox_area = obj.get("bbox_area", obj.get("bbox_area_px")) or 0

            stats["rows"] += 1
            stats["confidence_sum"] += float(depth_confidence)
            stats["confidence_count"] += 1

            history = histories[track_id]
            previous_smooth = _smooth(history)
            reasons = _should_accept(raw_m, previous_smooth, history, timestamp_ms, depth_quality)
            if float(depth_confidence) < DEPTH_MIN_CONFIDENCE_FOR_BASELINE:
                reasons.append("low_confidence_baseline")
            if int(bbox_area) < MIN_RELIABLE_BBOX_AREA_PX:
                reasons.append("bbox_too_small_for_reliable_depth")

            accepted = not reasons
            if (not accepted) and _finite(raw_m) and depth_quality == "good":
                candidates = rejected_candidates[track_id]
                candidates.append({"raw": float(raw_m), "conf": float(depth_confidence), "area": int(bbox_area)})
                if _recovery_ok(candidates, previous_smooth):
                    accepted = True
                    history.clear()
            elif accepted:
                rejected_candidates[track_id].clear()

            if accepted and _finite(raw_m):
                history.append((timestamp_ms, float(raw_m)))
            elif reasons:
                stats["rejected"] += 1

            smooth = _smooth(history)
            if smooth is None:
                stats["na"] += 1
            else:
                stats["valid"] += 1
                per_track = smooth_by_track[track_id]
                if per_track:
                    stats["max_jump"] = max(stats["max_jump"], abs(float(smooth) - per_track[-1]))
                per_track.append(float(smooth))

    closing_ok = 0
    closing_eligible = 0
    for values in smooth_by_track.values():
        if len(values) < 5:
            continue
        closing_eligible += 1
        if values[-1] <= values[0]:
            closing_ok += 1

    return {
        "percentile": f"p{percentile}",
        "valid_distance_frame_ratio": 0.0 if stats["rows"] == 0 else stats["valid"] / stats["rows"],
        "distance_na_count": stats["na"],
        "rejected_by_gate_count": stats["rejected"],
        "closing_tracks_down": f"{closing_ok}/{closing_eligible}",
        "max_single_frame_jump_m": stats["max_jump"],
        "avg_depth_confidence": 0.0
        if stats["confidence_count"] == 0
        else stats["confidence_sum"] / stats["confidence_count"],
        "track_interruptions": stats["track_interruptions"],
    }


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def print_table(rows):
    columns = [
        "percentile",
        "valid_distance_frame_ratio",
        "distance_na_count",
        "rejected_by_gate_count",
        "closing_tracks_down",
        "max_single_frame_jump_m",
        "avg_depth_confidence",
        "track_interruptions",
    ]
    print(",".join(columns))
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        print(",".join(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline_jsonl")
    args = parser.parse_args()

    records = load_records(args.timeline_jsonl)
    rows = [replay_percentile(records, percentile) for percentile in PERCENTILES]
    print_table(rows)


if __name__ == "__main__":
    main()
