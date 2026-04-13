# -*- coding: utf-8 -*-
"""Depth Anything / Depth Pro backend loaders and single-frame inference."""

import importlib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import DEPTH_ANYTHING_MODEL_ID, DEPTH_BACKEND, DEPTH_RESIZE_WIDTH
from ..paths import DEPTH_PRO_CHECKPOINT
from ..runtime import _ensure_depth_pro_import_path, _prepare_model_inputs, _resolve_runtime_settings


def _load_depth_anything_model(model_id=DEPTH_ANYTHING_MODEL_ID, runtime=None):
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
    model.train(False)

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
    model.train(False)

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
