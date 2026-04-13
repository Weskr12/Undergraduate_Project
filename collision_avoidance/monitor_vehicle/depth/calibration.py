# -*- coding: utf-8 -*-
"""Depth-to-meter calibration: fit linear/inverse mapping and apply it."""

import json
from pathlib import Path

import numpy as np

from ..config import MIN_DEPTH
from ..utils import to_float_or_none


def _extract_calibration_points(raw_points):
    points = []
    for item in raw_points:
        if isinstance(item, dict):
            raw_depth = float(item["raw_depth"])
            distance_m = float(item["distance_m"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            raw_depth = float(item[0])
            distance_m = float(item[1])
        else:
            raise ValueError(f"Unsupported calibration point format: {item}")
        points.append({"raw_depth": raw_depth, "distance_m": distance_m})
    return points


def _validate_calibration_points(points):
    if len(points) < 5:
        raise ValueError("Calibration requires at least 5 points")

    for idx, p in enumerate(points):
        raw_depth = p["raw_depth"]
        distance_m = p["distance_m"]
        if raw_depth <= 0 or distance_m <= 0:
            raise ValueError(f"Invalid calibration point #{idx}: {p}")


def _fit_linear(z_values, d_values):
    A = np.vstack([z_values, np.ones_like(z_values)]).T
    a, b = np.linalg.lstsq(A, d_values, rcond=None)[0]
    pred = a * z_values + b
    mae = float(np.mean(np.abs(pred - d_values)))
    return {
        "model_type": "linear",
        "a": float(a),
        "b": float(b),
        "mae": mae,
    }


def _fit_inverse(z_values, d_values, eps):
    inv_z = 1.0 / (z_values + eps)
    A = np.vstack([inv_z, np.ones_like(inv_z)]).T
    a, b = np.linalg.lstsq(A, d_values, rcond=None)[0]
    pred = a / (z_values + eps) + b
    mae = float(np.mean(np.abs(pred - d_values)))
    return {
        "model_type": "inverse",
        "a": float(a),
        "b": float(b),
        "eps": float(eps),
        "mae": mae,
    }


def load_calibration(calib_path):
    path = Path(calib_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}. Create a JSON file with 5 points under 'calibration_points'."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_points = data.get("calibration_points") or data.get("points")
    if raw_points is None:
        raise ValueError("Calibration JSON must contain 'calibration_points'")

    points = _extract_calibration_points(raw_points)
    _validate_calibration_points(points)

    z_values = np.array([p["raw_depth"] for p in points], dtype=np.float64)
    d_values = np.array([p["distance_m"] for p in points], dtype=np.float64)
    eps = float(data.get("eps", 1e-6))

    linear = _fit_linear(z_values, d_values)
    inverse = _fit_inverse(z_values, d_values, eps=eps)

    candidates = [linear, inverse]
    selected = min(candidates, key=lambda c: c["mae"])

    return {
        "source": str(path),
        "calibration_points": points,
        "candidates": candidates,
        "selected_model": selected,
    }


def map_depth_to_meters(raw_depth_value, calib_model):
    z = to_float_or_none(raw_depth_value)
    if z is None or z <= MIN_DEPTH:
        return None

    selected = calib_model.get("selected_model", calib_model)
    model_type = selected.get("model_type")
    a = float(selected["a"])
    b = float(selected["b"])

    if model_type == "linear":
        distance = a * z + b
    elif model_type == "inverse":
        eps = float(selected.get("eps", 1e-6))
        distance = a / (z + eps) + b
    else:
        raise ValueError(f"Unknown calibration model type: {model_type}")

    distance = to_float_or_none(distance)
    if distance is None:
        return None
    return max(0.0, distance)


def raw_depth_to_distance_m(raw_depth_value, depth_state, calib_model):
    units = (depth_state or {}).get("units", "relative")
    if units == "metric_m":
        z = to_float_or_none(raw_depth_value)
        if z is None or z <= MIN_DEPTH:
            return None
        return max(0.0, z)
    return map_depth_to_meters(raw_depth_value, calib_model)


def calibration_template():
    return {
        "calibration_points": [
            {"raw_depth": 4.2, "distance_m": 3.0},
            {"raw_depth": 3.4, "distance_m": 5.0},
            {"raw_depth": 2.7, "distance_m": 8.0},
            {"raw_depth": 2.1, "distance_m": 12.0},
            {"raw_depth": 1.8, "distance_m": 15.0},
        ]
    }
