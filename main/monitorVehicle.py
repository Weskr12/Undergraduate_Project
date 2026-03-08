# -*- coding: utf-8 -*-
# Auto-exported from monitorVehicle.ipynb

"""
monitorVehicle.py

用途:
離線執行車輛/機車偵測、追蹤、深度距離估測與右側雷達視覺化輸出。

執行前需求:
- YOLO 權重: model/weights/best.pt
- Depth Pro 權重: _tmp_ml_depth_pro/checkpoints/depth_pro.pt
- 測試影片: dataset/*.MP4

執行方式:
- 直接匯入後呼叫 run_mvp_pipeline(...)
- 或將檔案底部的 RUN_FULL_PIPELINE / RUN_SMOKE_TEST 改為 True 後執行

主要輸出:
- 標註影片: output/*.mp4
- 每幀時間線: output/*.jsonl
- 即時摘要: output/*.json

說明:
- 程式會優先使用 GPU(CUDA)；若無可用 GPU，則自動退回 CPU。
- 若使用非 metric depth backend，需提供 output/depth_calibration.json 做距離校正。
"""

# %% [code] cell 0
import importlib
import gc
import json
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

RUN_ID = f"{time.strftime('%Y%m%d_%H%M%S')}_test1_mvp"

YOLO_MODEL_PATH = Path("model/weights/best.pt")
DEPTH_BACKEND = "depth_pro"
DEPTH_ANYTHING_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_PRO_REPO_SRC = Path("_tmp_ml_depth_pro/src")
DEPTH_PRO_CHECKPOINT = Path("_tmp_ml_depth_pro/checkpoints/depth_pro.pt")
PREFERRED_DEVICE = "auto"
PREFER_HALF_PRECISION = True
YOLO_INFER_IMGSZ = 640

DEFAULT_VIDEO_PATH = Path("dataset/test1.MP4")
DEFAULT_OUTPUT_VIDEO = Path("output/test1_mvp_annotated.mp4")
DEFAULT_TIMELINE_JSONL = Path("output/radar_timeline.jsonl")
DEFAULT_LIVE_JSON = Path("output/radar_live.json")
DEFAULT_CALIB_PATH = Path("output/depth_calibration.json")

VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5]
HOOK_TURN_CLASS_ID = 5
TRACKER_PATH = "bytetrack.yaml"

CONF_THRESHOLDS = {
    0: 0.5,
    1: 0.5,
    2: 0.5,
    3: 0.5,
    4: 0.5,
    5: 0.1,
}

DEPTH_RESIZE_WIDTH = 640
DEPTH_INFER_EVERY_N = 2
DEPTH_PRO_CPU_INFER_EVERY_N = 4
TRACK_STALE_FRAMES = 60
DISTANCE_HISTORY_LEN = 8
SMOOTH_WINDOW = 5
APPROACH_LOOKBACK_FRAMES = 5
BBOX_EMA_ALPHA = 0.35
BEARING_EMA_ALPHA = 0.3
HFOV_DEG = 120.0
MIN_DEPTH = 1e-6
RADAR_PANEL_WIDTH = 320
RADAR_BEARING_CUTOFF_DEG = 20.0
RADAR_BANDS = ("far", "mid", "near")
RADAR_BAND_LABELS = {
    "far": "FAR",
    "mid": "MID",
    "near": "NEAR",
}
RADAR_MAX_DISTANCE_M = 35.0
RADAR_BG_COLOR = (20, 20, 20)
RADAR_PLOT_BG_COLOR = (34, 34, 34)
RADAR_BORDER_COLOR = (88, 88, 88)
RADAR_TEXT_COLOR = (235, 235, 235)
RADAR_SUBTEXT_COLOR = (180, 180, 180)
RADAR_OVERLAP_STEP_PX = 14
RADAR_OVERLAP_MARGIN_PX = 6

RISK_COLORS = {
    "critical": (0, 0, 255),
    "high": (0, 120, 255),
    "medium": (0, 220, 255),
    "low": (0, 200, 0),
    "unknown": (128, 128, 128),
    "info": (255, 0, 255),
}

RISK_SEVERITY = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "unknown": 0,
}


def ensure_parent(path_like):
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def to_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, (float, int)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    return None


def _ema(previous, current, alpha):
    if previous is None:
        return float(current)
    return float((1.0 - alpha) * previous + alpha * current)


def _bbox_xyxy_to_center_size(bbox_xyxy):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        max(1.0, x2 - x1),
        max(1.0, y2 - y1),
    )


def _center_size_to_bbox_xyxy(cx, cy, width, height, frame_w, frame_h):
    half_w = max(0.5, float(width) / 2.0)
    half_h = max(0.5, float(height) / 2.0)
    bbox = [
        cx - half_w,
        cy - half_h,
        cx + half_w,
        cy + half_h,
    ]
    return _clip_bbox_xyxy(bbox, frame_w, frame_h)


def _resolve_runtime_settings(preferred_device=PREFERRED_DEVICE):
    preferred = str(preferred_device or "auto").strip().lower()

    if preferred in {"", "auto"}:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but this PyTorch build does not have CUDA enabled.")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS device requested, but this PyTorch build does not support MPS.")

    cv2.setUseOptimized(True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    use_half = bool(PREFER_HALF_PRECISION and device.type == "cuda")
    precision = torch.float16 if use_half else torch.float32
    return {
        "torch_device": device,
        "device_str": str(device),
        "device_type": device.type,
        "use_half": use_half,
        "precision": precision,
    }


def _ensure_depth_pro_import_path():
    repo_src = DEPTH_PRO_REPO_SRC.resolve()
    if repo_src.exists():
        repo_src_str = str(repo_src)
        if repo_src_str not in sys.path:
            sys.path.insert(0, repo_src_str)


print("MVP constants loaded")

# %% [code] cell 1
def _prepare_model_inputs(inputs, device, use_half=False):
    prepared = {}
    for key, value in inputs.items():
        tensor = value.to(device)
        if use_half and torch.is_floating_point(tensor):
            tensor = tensor.half()
        prepared[key] = tensor
    return prepared


def _load_depth_anything_model(model_id=DEPTH_ANYTHING_MODEL_ID, runtime=None):
    # Load Depth Anything model. Fallback to local cache if network is blocked.
    runtime = runtime or _resolve_runtime_settings()
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Depth Anything. Install with: pip install transformers accelerate"
        ) from exc

    load_kwargs = {
        "trust_remote_code": False,
    }

    try:
        processor = AutoImageProcessor.from_pretrained(model_id, **load_kwargs)
        model = AutoModelForDepthEstimation.from_pretrained(model_id, **load_kwargs)
        load_mode = "remote_or_cache"
    except Exception:
        # Use local cache only if remote lookup is blocked.
        local_kwargs = dict(load_kwargs)
        local_kwargs["local_files_only"] = True
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, **local_kwargs)
            model = AutoModelForDepthEstimation.from_pretrained(model_id, **local_kwargs)
            load_mode = "local_cache_only"
        except Exception as inner_exc:
            raise RuntimeError(
                "Cannot load Depth Anything model. Ensure internet access for first download or pre-cache the model locally."
            ) from inner_exc

    model.to(runtime["torch_device"])
    if runtime["use_half"]:
        model.half()
    model.eval()

    return {
        "backend": "depth_anything",
        "units": "relative",
        "model_id": model_id,
        "device": runtime["device_str"],
        "use_half": runtime["use_half"],
        "processor": processor,
        "model": model,
        "load_mode": load_mode,
    }


def _load_depth_pro_model(checkpoint_path=DEPTH_PRO_CHECKPOINT, runtime=None):
    runtime = runtime or _resolve_runtime_settings()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Depth Pro checkpoint not found: {checkpoint_path}. "
            "Install the model and download depth_pro.pt first."
        )

    try:
        _ensure_depth_pro_import_path()
        depth_pro_module = importlib.import_module("depth_pro.depth_pro")
    except ImportError as exc:
        raise RuntimeError(
            "depth_pro is required for the Depth Pro backend. Install the official package first."
        ) from exc

    default_config = depth_pro_module.DEFAULT_MONODEPTH_CONFIG_DICT
    create_model_and_transforms = depth_pro_module.create_model_and_transforms
    config = replace(default_config, checkpoint_uri=str(checkpoint_path.resolve()))
    model, transform = create_model_and_transforms(
        config=config,
        device=runtime["torch_device"],
        precision=runtime["precision"],
    )
    model.eval()

    return {
        "backend": "depth_pro",
        "units": "metric_m",
        "device": runtime["device_str"],
        "use_half": runtime["use_half"],
        "model": model,
        "transform": transform,
        "checkpoint_path": str(checkpoint_path),
    }


def load_depth_model(backend=DEPTH_BACKEND, model_id=DEPTH_ANYTHING_MODEL_ID, runtime=None):
    runtime = runtime or _resolve_runtime_settings()
    backend = str(backend).lower()
    if backend == "depth_anything":
        return _load_depth_anything_model(model_id=model_id, runtime=runtime)
    if backend == "depth_pro":
        return _load_depth_pro_model(runtime=runtime)
    raise ValueError(f"Unsupported depth backend: {backend}")


def _infer_depth_anything(frame_bgr, depth_state):
    processor = depth_state["processor"]
    model = depth_state["model"]
    device = depth_state["device"]
    use_half = bool(depth_state.get("use_half"))

    h, w = frame_bgr.shape[:2]
    if w > DEPTH_RESIZE_WIDTH:
        scale = DEPTH_RESIZE_WIDTH / float(w)
        resized = cv2.resize(
            frame_bgr,
            (DEPTH_RESIZE_WIDTH, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = frame_bgr

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = _prepare_model_inputs(inputs, device=device, use_half=use_half)

    with torch.inference_mode():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)

    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


def _infer_depth_pro(frame_bgr, depth_state):
    from PIL import Image

    model = depth_state["model"]
    transform = depth_state["transform"]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = transform(image)

    with torch.inference_mode():
        prediction = model.infer(inputs, f_px=None)
        depth = prediction["depth"].detach().cpu().numpy().astype(np.float32)

    depth = np.squeeze(depth)
    if depth.shape[:2] != frame_bgr.shape[:2]:
        depth = cv2.resize(depth, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth


def infer_depth_map(frame_bgr, depth_state):
    # Infer raw/metric depth map for one BGR frame.
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame_bgr is empty")
    if depth_state is None:
        raise ValueError("depth_state is required")

    backend = depth_state.get("backend")
    if backend == "depth_anything":
        depth = _infer_depth_anything(frame_bgr, depth_state)
    elif backend == "depth_pro":
        depth = _infer_depth_pro(frame_bgr, depth_state)
    else:
        raise ValueError(f"Unknown depth backend: {backend}")

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return depth

# %% [code] cell 2
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
    # Load calibration points and auto-select best mapping model by MAE.
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
    # Map raw depth to distance in meters using selected calibration model.
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

# %% [code] cell 3
def _clip_bbox_xyxy(bbox_xyxy, frame_w, frame_h):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w - 1))
    y2 = max(0, min(y2, frame_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def compute_object_distance(depth_map, bbox_xyxy):
    # Use bbox bottom 30% and middle 60% width, then take p35 depth.
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


def _extract_detections_from_boxes(boxes, model_names, frame_shape, next_temp_track_id, force_temp_ids=False):
    frame_h, frame_w = frame_shape[:2]
    detections = []

    if boxes is None or len(boxes) == 0:
        return detections, next_temp_track_id

    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy().astype(float)

    track_ids = None
    if (not force_temp_ids) and getattr(boxes, "id", None) is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)

    for idx, bbox in enumerate(xyxy):
        class_id = int(clss[idx])
        conf = float(confs[idx])

        if conf < CONF_THRESHOLDS.get(class_id, 0.5):
            continue

        clipped_bbox = _clip_bbox_xyxy(bbox.tolist(), frame_w, frame_h)
        if clipped_bbox is None:
            continue

        if track_ids is None:
            track_id = next_temp_track_id
            next_temp_track_id -= 1
        else:
            track_id = int(track_ids[idx])

        class_name = model_names.get(class_id, str(class_id)) if isinstance(model_names, dict) else str(class_id)

        detections.append(
            {
                "track_id": int(track_id),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": clipped_bbox,
            }
        )

    return detections, next_temp_track_id


def _detect_with_tracking(frame_bgr, yolo_model, next_temp_track_id, runtime):
    track_exception = None
    infer_kwargs = {
        "classes": VEHICLE_CLASSES,
        "conf": 0.1,
        "device": runtime["device_str"],
        "imgsz": YOLO_INFER_IMGSZ,
        "verbose": False,
    }
    if runtime["use_half"]:
        infer_kwargs["half"] = True

    try:
        track_results = yolo_model.track(
            frame_bgr,
            persist=True,
            tracker=TRACKER_PATH,
            **infer_kwargs,
        )
    except Exception as exc:
        track_results = None
        track_exception = exc

    if track_results and len(track_results) > 0:
        track_boxes = track_results[0].boxes
        if track_boxes is not None and len(track_boxes) > 0 and getattr(track_boxes, "id", None) is not None:
            detections, next_temp_track_id = _extract_detections_from_boxes(
                track_boxes,
                yolo_model.names,
                frame_bgr.shape,
                next_temp_track_id,
                force_temp_ids=False,
            )
            return detections, next_temp_track_id, "track", track_exception

    detect_results = yolo_model(frame_bgr, **infer_kwargs)
    detect_boxes = detect_results[0].boxes if detect_results and len(detect_results) > 0 else None

    detections, next_temp_track_id = _extract_detections_from_boxes(
        detect_boxes,
        yolo_model.names,
        frame_bgr.shape,
        next_temp_track_id,
        force_temp_ids=True,
    )
    return detections, next_temp_track_id, "fallback_detect", track_exception

# %% [code] cell 4
def _distance_band(distance_m):
    if distance_m is None:
        return "unknown"
    if distance_m <= 8.0:
        return "near"
    if distance_m <= 15.0:
        return "mid"
    return "far"


def _risk_level(distance_m, approach_mps):
    if distance_m is None:
        return "unknown"
    if distance_m <= 4.0:
        return "critical"
    if distance_m <= 8.0 or (distance_m <= 10.0 and approach_mps >= 1.5):
        return "high"
    if distance_m <= 15.0 or approach_mps >= 0.8:
        return "medium"
    return "low"


def _bbox_bearing_deg(bbox_xyxy, frame_w, hfov_deg=HFOV_DEG):
    x1, _, x2, _ = bbox_xyxy
    center_x = 0.5 * (x1 + x2)
    normalized = ((center_x / float(frame_w)) - 0.5) * 2.0
    bearing = normalized * (hfov_deg / 2.0)
    return float(np.clip(bearing, -hfov_deg / 2.0, hfov_deg / 2.0))


def _smooth_detection_display(obj, display_track_state, frame_w, frame_h):
    track_id = int(obj.get("track_id", -1))
    if track_id < 0:
        return obj

    bbox = obj.get("bbox_xyxy")
    bearing_deg = to_float_or_none(obj.get("bearing_deg"))
    if bbox is None or len(bbox) != 4:
        return obj

    center_x, center_y, width, height = _bbox_xyxy_to_center_size(bbox)
    state = display_track_state.get(track_id, {})

    smooth_center_x = _ema(state.get("center_x"), center_x, BBOX_EMA_ALPHA)
    smooth_center_y = _ema(state.get("center_y"), center_y, BBOX_EMA_ALPHA)
    smooth_width = _ema(state.get("width"), width, BBOX_EMA_ALPHA)
    smooth_height = _ema(state.get("height"), height, BBOX_EMA_ALPHA)
    smooth_bearing = bearing_deg
    if bearing_deg is not None:
        smooth_bearing = _ema(state.get("bearing_deg"), bearing_deg, BEARING_EMA_ALPHA)

    smoothed_bbox = _center_size_to_bbox_xyxy(
        smooth_center_x,
        smooth_center_y,
        smooth_width,
        smooth_height,
        frame_w,
        frame_h,
    )
    if smoothed_bbox is not None:
        obj["bbox_xyxy"] = smoothed_bbox
    if smooth_bearing is not None:
        obj["bearing_deg"] = round(float(smooth_bearing), 2)

    display_track_state[track_id] = {
        "center_x": smooth_center_x,
        "center_y": smooth_center_y,
        "width": smooth_width,
        "height": smooth_height,
        "bearing_deg": smooth_bearing,
    }
    return obj


def build_radar_payload(frame_idx, timestamp_ms, detections, meta):
    # Build one frame payload for live JSON and timeline JSONL.
    valid_distances = [d["distance_m"] for d in detections if d.get("distance_m") is not None]
    high_risk_count = sum(1 for d in detections if d.get("risk_level") in {"high", "critical"})
    hook_turn_detected = any(d.get("hook_turn_detected", False) for d in detections)

    payload = {
        "run_id": meta["run_id"],
        "frame_idx": int(frame_idx),
        "timestamp_ms": int(timestamp_ms),
        "fps_source": float(meta["fps_source"]),
        "objects": detections,
        "summary": {
            "object_count": int(len(detections)),
            "min_distance_m": float(min(valid_distances)) if valid_distances else None,
            "high_risk_count": int(high_risk_count),
            "hook_turn_detected": bool(hook_turn_detected),
        },
    }
    return payload


def write_live_json(payload, live_json_path):
    path = ensure_parent(live_json_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _append_timeline_jsonl(payload, timeline_jsonl_path):
    path = ensure_parent(timeline_jsonl_path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _short_class_name(class_name):
    name = str(class_name).lower()
    alias = {
        "motorcycle": "moto",
        "bicycle": "bike",
        "person": "person",
        "truck": "truck",
        "bus": "bus",
        "car": "car",
    }
    return alias.get(name, name[:8])


def _iter_radar_detections(detections):
    radar_detections = []

    for det in detections:
        if det.get("hook_turn_detected"):
            continue
        distance_m = to_float_or_none(det.get("distance_m"))
        bearing_deg = to_float_or_none(det.get("bearing_deg"))
        if distance_m is None or bearing_deg is None:
            continue
        radar_detections.append(det)

    return radar_detections


def _distance_to_plot_fraction(distance_m):
    distance = max(0.0, to_float_or_none(distance_m) or 0.0)

    if distance <= 8.0:
        progress = np.clip(distance / 8.0, 0.0, 1.0)
        return float(1.0 - progress * (1.0 / 3.0))
    if distance <= 15.0:
        progress = np.clip((distance - 8.0) / 7.0, 0.0, 1.0)
        return float((2.0 / 3.0) - progress * (1.0 / 3.0))

    progress = np.clip((distance - 15.0) / max(RADAR_MAX_DISTANCE_M - 15.0, 1e-6), 0.0, 1.0)
    return float((1.0 / 3.0) - progress * (1.0 / 3.0))


def _radar_target_position(det, plot_left, plot_top, plot_right, plot_bottom):
    distance_m = to_float_or_none(det.get("distance_m"))
    bearing_deg = to_float_or_none(det.get("bearing_deg"))
    if distance_m is None or bearing_deg is None:
        return None

    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)
    usable_hfov = max(HFOV_DEG, 1.0)

    x_norm = np.clip((bearing_deg + usable_hfov / 2.0) / usable_hfov, 0.0, 1.0)
    y_norm = np.clip(_distance_to_plot_fraction(distance_m), 0.0, 1.0)

    jitter_seed = abs(int(det.get("track_id", 0)))
    jitter_x = ((jitter_seed % 5) - 2) * 3
    jitter_y = (((jitter_seed // 5) % 5) - 2) * 2

    x = int(round(plot_left + x_norm * plot_width + jitter_x))
    y = int(round(plot_top + y_norm * plot_height + jitter_y))
    x = int(np.clip(x, plot_left + 8, plot_right - 8))
    y = int(np.clip(y, plot_top + 8, plot_bottom - 8))
    return (x, y)


def _clamp_radar_point(x, y, radius, plot_left, plot_top, plot_right, plot_bottom):
    clamped_x = int(np.clip(x, plot_left + radius + 2, plot_right - radius - 2))
    clamped_y = int(np.clip(y, plot_top + radius + 2, plot_bottom - radius - 2))
    return clamped_x, clamped_y


def _radar_overlap_offsets():
    base_offsets = [
        (0, 0),
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ]
    for ring in range(1, 5):
        for offset_x, offset_y in base_offsets:
            yield (
                offset_x * ring * RADAR_OVERLAP_STEP_PX,
                offset_y * ring * RADAR_OVERLAP_STEP_PX,
            )


def _find_non_overlapping_radar_position(
    base_position,
    radius,
    placed_targets,
    plot_left,
    plot_top,
    plot_right,
    plot_bottom,
):
    if base_position is None:
        return None

    for offset_x, offset_y in _radar_overlap_offsets():
        candidate_x, candidate_y = _clamp_radar_point(
            base_position[0] + offset_x,
            base_position[1] + offset_y,
            radius,
            plot_left,
            plot_top,
            plot_right,
            plot_bottom,
        )
        overlaps = False
        for placed in placed_targets:
            min_spacing = radius + placed["radius"] + RADAR_OVERLAP_MARGIN_PX
            if math.hypot(candidate_x - placed["x"], candidate_y - placed["y"]) < min_spacing:
                overlaps = True
                break
        if not overlaps:
            return (candidate_x, candidate_y)

    return _clamp_radar_point(
        base_position[0],
        base_position[1],
        radius,
        plot_left,
        plot_top,
        plot_right,
        plot_bottom,
    )


def _radar_target_radius(det):
    risk_level = det.get("risk_level", "unknown")
    distance_m = to_float_or_none(det.get("distance_m")) or RADAR_MAX_DISTANCE_M
    if risk_level in {"critical", "high"}:
        return 10
    if distance_m <= 8.0:
        return 9
    return 7


def _select_radar_list_targets(radar_detections, max_items=6):
    return sorted(
        radar_detections,
        key=lambda det: (
            to_float_or_none(det.get("distance_m")) if det.get("distance_m") is not None else float("inf"),
            -RISK_SEVERITY.get(det.get("risk_level", "unknown"), 0),
            abs(to_float_or_none(det.get("bearing_deg")) or 0.0),
            int(det.get("track_id", 0)),
        ),
    )[:max_items]


def _bearing_label(bearing_deg):
    bearing = to_float_or_none(bearing_deg) or 0.0
    if bearing < -RADAR_BEARING_CUTOFF_DEG:
        return "L"
    if bearing > RADAR_BEARING_CUTOFF_DEG:
        return "R"
    return "C"


def _draw_radar_guides(panel, plot_left, plot_top, plot_right, plot_bottom):
    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), RADAR_PLOT_BG_COLOR, -1)
    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), RADAR_BORDER_COLOR, 1)

    plot_height = plot_bottom - plot_top
    plot_width = plot_right - plot_left
    far_y = plot_top + int(round(plot_height / 3.0))
    mid_y = plot_top + int(round(plot_height * 2.0 / 3.0))
    center_x = plot_left + plot_width // 2

    cv2.line(panel, (center_x, plot_top), (center_x, plot_bottom), RADAR_BORDER_COLOR, 1)
    cv2.line(panel, (plot_left, far_y), (plot_right, far_y), RADAR_BORDER_COLOR, 1)
    cv2.line(panel, (plot_left, mid_y), (plot_right, mid_y), RADAR_BORDER_COLOR, 1)

    cv2.putText(panel, "L", (plot_left + 2, plot_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_SUBTEXT_COLOR, 1)
    cv2.putText(panel, "C", (center_x - 6, plot_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_SUBTEXT_COLOR, 1)
    cv2.putText(panel, "R", (plot_right - 12, plot_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_SUBTEXT_COLOR, 1)

    band_positions = {
        "far": plot_top + 22,
        "mid": far_y + 22,
        "near": mid_y + 22,
    }
    for band in RADAR_BANDS:
        cv2.putText(
            panel,
            RADAR_BAND_LABELS[band],
            (plot_left + 8, band_positions[band]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            RADAR_SUBTEXT_COLOR,
            1,
        )


def _draw_radar_targets(panel, radar_detections, plot_left, plot_top, plot_right, plot_bottom):
    sorted_targets = sorted(
        radar_detections,
        key=lambda det: (
            -RISK_SEVERITY.get(det.get("risk_level", "unknown"), 0),
            to_float_or_none(det.get("distance_m")) if det.get("distance_m") is not None else float("inf"),
            abs(to_float_or_none(det.get("bearing_deg")) or 0.0),
        ),
    )
    placed_targets = []

    for det in sorted_targets:
        base_position = _radar_target_position(det, plot_left, plot_top, plot_right, plot_bottom)
        radius = _radar_target_radius(det)
        position = _find_non_overlapping_radar_position(
            base_position,
            radius,
            placed_targets,
            plot_left,
            plot_top,
            plot_right,
            plot_bottom,
        )
        if position is None:
            continue

        x, y = position
        color = RISK_COLORS.get(det.get("risk_level"), RISK_COLORS["unknown"])
        cv2.circle(panel, (x, y), radius + 2, (235, 235, 235), -1)
        cv2.circle(panel, (x, y), radius, color, -1)
        placed_targets.append({"x": x, "y": y, "radius": radius})

        distance_m = to_float_or_none(det.get("distance_m")) or 0.0
        risk_level = det.get("risk_level", "unknown")
        if risk_level in {"critical", "high"} or distance_m <= 8.0:
            label = f"{_short_class_name(det['class_name'])} {distance_m:.1f}m"
            text_x = int(np.clip(x + 10, plot_left + 6, plot_right - 96))
            text_y = int(np.clip(y - 8, plot_top + 16, plot_bottom - 4))
            cv2.putText(panel, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RADAR_TEXT_COLOR, 1)


def _draw_radar_list(panel, radar_detections, panel_h):
    list_targets = _select_radar_list_targets(radar_detections)
    list_top = panel_h - 142

    cv2.putText(panel, "Closest", (16, list_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RADAR_TEXT_COLOR, 1)
    row_y = list_top + 24

    for det in list_targets:
        color = RISK_COLORS.get(det.get("risk_level"), RISK_COLORS["unknown"])
        distance_m = to_float_or_none(det.get("distance_m")) or 0.0
        bearing_label = _bearing_label(det.get("bearing_deg"))
        text = f"{bearing_label} {_short_class_name(det['class_name'])} {distance_m:>4.1f}m"
        cv2.circle(panel, (20, row_y - 5), 5, color, -1)
        cv2.putText(panel, text, (32, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_TEXT_COLOR, 1)
        row_y += 20

    if not list_targets:
        cv2.putText(panel, "no valid targets", (16, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_SUBTEXT_COLOR, 1)


def _draw_radar_panel(output_frame, detections):
    frame_h, frame_w = output_frame.shape[:2]
    panel_x0 = frame_w - RADAR_PANEL_WIDTH
    panel = output_frame[:, panel_x0:frame_w]
    panel[:] = RADAR_BG_COLOR

    radar_detections = _iter_radar_detections(detections)
    min_distance = min((det["distance_m"] for det in radar_detections), default=None)

    cv2.putText(panel, "Radar", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RADAR_TEXT_COLOR, 2)
    cv2.putText(
        panel,
        f"objects {len(radar_detections)}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        RADAR_SUBTEXT_COLOR,
        1,
    )
    min_distance_txt = "min n/a" if min_distance is None else f"min {min_distance:.1f}m"
    cv2.putText(
        panel,
        min_distance_txt,
        (164, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        RADAR_SUBTEXT_COLOR,
        1,
    )

    plot_left = 18
    plot_right = RADAR_PANEL_WIDTH - 18
    plot_top = 96
    plot_bottom = frame_h - 176

    _draw_radar_guides(panel, plot_left, plot_top, plot_right, plot_bottom)
    _draw_radar_targets(panel, radar_detections, plot_left, plot_top, plot_right, plot_bottom)
    _draw_radar_list(panel, radar_detections, frame_h)


def _compose_output_frame(frame_bgr, detections):
    frame_h, frame_w = frame_bgr.shape[:2]
    canvas = np.zeros((frame_h, frame_w + RADAR_PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :frame_w] = frame_bgr
    _draw_radar_panel(canvas, detections)
    return canvas


def _draw_detection(frame_bgr, det):
    x1, y1, x2, y2 = det["bbox_xyxy"]
    risk = det["risk_level"]
    color = RISK_COLORS.get(risk, RISK_COLORS["unknown"])

    if det["hook_turn_detected"]:
        color = RISK_COLORS["info"]

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    dist_txt = "n/a" if det["distance_m"] is None else f"{det['distance_m']:.1f}m"
    label = (
        f"ID:{det['track_id']} {det['class_name']} "
        f"{dist_txt} {det['risk_level']} b:{det['bearing_deg']:.1f}"
    )
    cv2.putText(
        frame_bgr,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
    )
# %% [code] cell 5
def run_mvp_pipeline(
    video_path,
    output_video_path,
    timeline_jsonl_path,
    live_json_path,
    calib_path,
    max_frames=None,
):
    # Run full offline MVP pipeline and generate annotated video + JSON outputs.
    video_path = Path(video_path)
    output_video_path = ensure_parent(output_video_path)
    timeline_jsonl_path = ensure_parent(timeline_jsonl_path)
    live_json_path = ensure_parent(live_json_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w + RADAR_PANEL_WIDTH, frame_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_video_path}")

    timeline_jsonl_path.write_text("", encoding="utf-8")

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

    distance_history = defaultdict(lambda: deque(maxlen=DISTANCE_HISTORY_LEN))
    display_track_state = {}
    last_seen_frame = {}
    next_temp_track_id = -1

    last_depth_map = None

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

        should_run_depth = (frame_idx % depth_infer_every_n == 0) or (last_depth_map is None)
        depth_stale = not should_run_depth

        if should_run_depth:
            try:
                last_depth_map = infer_depth_map(frame, depth_state)
                depth_stale = False
            except Exception:
                depth_stale = True

        detections, next_temp_track_id, mode_used, track_exception = _detect_with_tracking(
            frame,
            yolo_model,
            next_temp_track_id,
            runtime,
        )

        objects = []
        hook_turn_detected = False

        for det in detections:
            track_id = int(det["track_id"])
            class_id = int(det["class_id"])
            class_name = str(det["class_name"])
            confidence = float(det["confidence"])
            bbox = [int(v) for v in det["bbox_xyxy"]]

            is_hook_turn = class_id == HOOK_TURN_CLASS_ID
            hook_turn_detected = hook_turn_detected or is_hook_turn

            bearing_deg = _bbox_bearing_deg(bbox, frame_w)
            raw_depth = None
            distance_m = None
            approach_mps = 0.0
            distance_band = "unknown"
            risk_level = "unknown"

            if not is_hook_turn and last_depth_map is not None:
                raw_depth = compute_object_distance(last_depth_map, bbox)
                distance_m = raw_depth_to_distance_m(raw_depth, depth_state, calib_model)

                if distance_m is not None:
                    history = distance_history[track_id]
                    history.append((timestamp_ms, float(distance_m)))

                    history_items = list(history)
                    smooth_values = [v for _, v in history_items[-SMOOTH_WINDOW:]]
                    distance_m = float(np.median(smooth_values))

                    if len(history_items) > APPROACH_LOOKBACK_FRAMES:
                        old_ts, old_distance = history_items[-1 - APPROACH_LOOKBACK_FRAMES]
                        dt = max((timestamp_ms - old_ts) / 1000.0, 1e-6)
                        approach_mps = max(0.0, (old_distance - distance_m) / dt)

                distance_band = _distance_band(distance_m)
                risk_level = _risk_level(distance_m, approach_mps)
            elif is_hook_turn:
                risk_level = "low"

            if track_id >= 0:
                last_seen_frame[track_id] = frame_idx

            if not is_hook_turn:
                detected_objects += 1
                if distance_m is not None:
                    valid_distance_objects += 1

            obj = {
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": bbox,
                "bearing_deg": round(float(bearing_deg), 2),
                "distance_m": None if distance_m is None else round(float(distance_m), 3),
                "distance_band": distance_band,
                "approach_mps": round(float(approach_mps), 3),
                "risk_level": risk_level,
                "hook_turn_detected": bool(is_hook_turn),
                "depth_stale": bool(depth_stale),
            }
            _smooth_detection_display(obj, display_track_state, frame_w, frame_h)
            objects.append(obj)
            _draw_detection(frame, obj)

        stale_track_ids = [
            tid
            for tid, last_frame in last_seen_frame.items()
            if (frame_idx - last_frame) > TRACK_STALE_FRAMES
        ]
        for tid in stale_track_ids:
            last_seen_frame.pop(tid, None)
            distance_history.pop(tid, None)
            display_track_state.pop(tid, None)

        payload = build_radar_payload(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            detections=objects,
            meta={"run_id": RUN_ID, "fps_source": fps},
        )

        if track_exception is not None:
            payload["summary"]["track_warning"] = str(track_exception)
        payload["summary"]["mode_used"] = mode_used
        payload["summary"]["hook_turn_detected"] = bool(hook_turn_detected)

        write_live_json(payload, live_json_path)
        _append_timeline_jsonl(payload, timeline_jsonl_path)

        min_distance = payload["summary"]["min_distance_m"]
        min_distance_txt = "n/a" if min_distance is None else f"{min_distance:.1f}m"
        info_text = (
            f"Frame:{frame_idx} Objects:{payload['summary']['object_count']} "
            f"MinDist:{min_distance_txt}"
        )
        cv2.putText(frame, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        output_frame = _compose_output_frame(frame, objects)
        writer.write(output_frame)
        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = max(time.time() - start_ts, 1e-6)
    avg_proc_fps = frame_idx / elapsed
    distance_valid_ratio = (valid_distance_objects / detected_objects) if detected_objects else 0.0
    depth_backend = depth_state.get("backend")
    runtime_device = runtime["device_str"]
    half_precision = bool(runtime["use_half"])
    selected_calibration_model = None if calib_model is None else calib_model["selected_model"]

    last_depth_map = None
    del yolo_model
    del depth_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_id": RUN_ID,
        "video_path": str(video_path),
        "frames_processed": int(frame_idx),
        "source_fps": float(fps),
        "processing_seconds": float(round(elapsed, 3)),
        "avg_processing_fps": float(round(avg_proc_fps, 3)),
        "distance_valid_ratio": float(round(distance_valid_ratio, 4)),
        "depth_backend": depth_backend,
        "runtime_device": runtime_device,
        "half_precision": half_precision,
        "depth_infer_every_n": int(depth_infer_every_n),
        "selected_calibration_model": selected_calibration_model,
        "outputs": {
            "annotated_video": str(output_video_path),
            "timeline_jsonl": str(timeline_jsonl_path),
            "live_json": str(live_json_path),
        },
    }

# %% [code] cell 6
def _validate_payload_schema(payload):
    top_keys = {"run_id", "frame_idx", "timestamp_ms", "fps_source", "objects", "summary"}
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
        "risk_level",
        "hook_turn_detected",
        "depth_stale",
    }

    assert top_keys.issubset(payload.keys()), f"Missing top-level keys: {top_keys - set(payload.keys())}"
    for obj in payload["objects"]:
        assert object_keys.issubset(obj.keys()), f"Missing object keys: {object_keys - set(obj.keys())}"


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


if DEPTH_BACKEND != "depth_pro" and not DEFAULT_CALIB_PATH.exists():
    print("Calibration file not found:", DEFAULT_CALIB_PATH)
    print("Create it with 5 measured points, for example:")
    print(json.dumps(calibration_template(), ensure_ascii=False, indent=2))

RUN_SMOKE_TEST = False
RUN_FULL_PIPELINE = False

if RUN_SMOKE_TEST:
    smoke_summary = run_mvp_pipeline(
        video_path=DEFAULT_VIDEO_PATH,
        output_video_path=DEFAULT_OUTPUT_VIDEO,
        timeline_jsonl_path=DEFAULT_TIMELINE_JSONL,
        live_json_path=DEFAULT_LIVE_JSON,
        calib_path=DEFAULT_CALIB_PATH,
        max_frames=100,
    )

    print("Smoke summary:")
    print(json.dumps(smoke_summary, ensure_ascii=False, indent=2))

    lines = DEFAULT_TIMELINE_JSONL.read_text(encoding="utf-8").splitlines()
    assert len(lines) == smoke_summary["frames_processed"], "Timeline JSONL line count mismatch"

    first_payload = json.loads(lines[0])
    _validate_payload_schema(first_payload)
    json.loads(DEFAULT_LIVE_JSON.read_text(encoding="utf-8"))
    print("Smoke test passed")

if RUN_FULL_PIPELINE:
    full_summary = run_mvp_pipeline(
        video_path=DEFAULT_VIDEO_PATH,
        output_video_path=DEFAULT_OUTPUT_VIDEO,
        timeline_jsonl_path=DEFAULT_TIMELINE_JSONL,
        live_json_path=DEFAULT_LIVE_JSON,
        calib_path=DEFAULT_CALIB_PATH,
    )
    print("Full run summary:")
    print(json.dumps(full_summary, ensure_ascii=False, indent=2))

