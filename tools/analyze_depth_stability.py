# -*- coding: utf-8 -*-
"""Summarize depth stability metrics from monitor_vehicle JSONL output."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pstdev(values)


def _mean_abs_delta(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    deltas = [abs(values[idx] - values[idx - 1]) for idx in range(1, len(values))]
    return _mean(deltas)


def _max_abs_delta(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return max(abs(values[idx] - values[idx - 1]) for idx in range(1, len(values)))


def _band_flip_count(bands):
    bands = [band for band in bands if band not in {None, "", "unknown"}]
    if len(bands) < 2:
        return 0
    return sum(1 for idx in range(1, len(bands)) if bands[idx] != bands[idx - 1])


def load_tracks(jsonl_path):
    tracks = defaultdict(list)
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            frame_idx = payload.get("frame_idx")
            for obj in payload.get("objects", []):
                track_id = obj.get("track_id")
                if track_id is None:
                    continue
                tracks[int(track_id)].append(
                    {
                        "frame_idx": frame_idx,
                        "raw": _to_float(obj.get("depth_raw_m")),
                        "smooth": _to_float(obj.get("depth_smooth_m")),
                        "used": _to_float(obj.get("depth_used_m", obj.get("distance_m"))),
                        "is_outlier": bool(obj.get("is_outlier", False)),
                        "is_held": obj.get("track_status") == "held" or int(obj.get("missing_frames", 0) or 0) > 0,
                        "distance_band": obj.get("distance_band"),
                        "confidence": _to_float(obj.get("confidence")),
                    }
                )
    return tracks


def summarize_tracks(tracks):
    rows = []
    for track_id, items in sorted(tracks.items()):
        raw_values = [item["raw"] for item in items]
        smooth_values = [item["smooth"] for item in items]
        used_values = [item["used"] for item in items]
        outlier_count = sum(1 for item in items if item["is_outlier"])
        held_count = sum(1 for item in items if item["is_held"])
        frame_count = len(items)
        rows.append(
            {
                "track_id": track_id,
                "frame_count": frame_count,
                "raw_depth_std": _std(raw_values),
                "smooth_depth_std": _std(smooth_values),
                "used_depth_std": _std(used_values),
                "raw_delta_mean": _mean_abs_delta(raw_values),
                "used_delta_mean": _mean_abs_delta(used_values),
                "max_jump": _max_abs_delta(used_values),
                "outlier_count": outlier_count,
                "outlier_rate": outlier_count / max(frame_count, 1),
                "missing_held_count": held_count,
                "distance_band_flip_count": _band_flip_count([item["distance_band"] for item in items]),
                "avg_confidence": _mean([item["confidence"] for item in items]),
            }
        )
    return rows


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(rows):
    fields = [
        "track_id",
        "frame_count",
        "raw_depth_std",
        "smooth_depth_std",
        "used_depth_std",
        "raw_delta_mean",
        "used_delta_mean",
        "max_jump",
        "outlier_count",
        "outlier_rate",
        "missing_held_count",
        "distance_band_flip_count",
        "avg_confidence",
    ]
    print(",".join(fields))
    for row in rows:
        print(",".join(_format_value(row.get(field)) for field in fields))


def write_csv(rows, output_path):
    fields = list(rows[0].keys()) if rows else [
        "track_id",
        "frame_count",
        "raw_depth_std",
        "smooth_depth_std",
        "used_depth_std",
        "raw_delta_mean",
        "used_delta_mean",
        "max_jump",
        "outlier_count",
        "outlier_rate",
        "missing_held_count",
        "distance_band_flip_count",
        "avg_confidence",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze per-track depth stability from JSONL.")
    parser.add_argument("jsonl_path", help="Path to *_radar_timeline.jsonl")
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV output path")
    args = parser.parse_args()

    tracks = load_tracks(args.jsonl_path)
    rows = summarize_tracks(tracks)
    if args.csv_path:
        write_csv(rows, args.csv_path)
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
