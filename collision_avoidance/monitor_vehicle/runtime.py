# -*- coding: utf-8 -*-
"""Torch/OpenCV runtime setup and depth_pro sys.path injection."""

import sys

import cv2
import torch

from .config import PREFER_HALF_PRECISION, PREFERRED_DEVICE
from .paths import DEPTH_PRO_REPO_SRC


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


def _prepare_model_inputs(inputs, device, use_half=False):
    prepared = {}
    for key, value in inputs.items():
        tensor = value.to(device)
        if use_half and torch.is_floating_point(tensor):
            tensor = tensor.half()
        prepared[key] = tensor
    return prepared


def _ensure_depth_pro_import_path():
    repo_src = DEPTH_PRO_REPO_SRC.resolve()
    if repo_src.exists():
        repo_src_str = str(repo_src)
        if repo_src_str not in sys.path:
            sys.path.insert(0, repo_src_str)
