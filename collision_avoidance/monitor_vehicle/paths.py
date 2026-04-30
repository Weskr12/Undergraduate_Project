# -*- coding: utf-8 -*-
"""Project path discovery and derived default paths.

Importing this module triggers path discovery and RUN_ID generation, matching
the original monitorVehicle.py import-time behavior.
"""

import time
from pathlib import Path

from .config import INPUT_VIDEO_NAME, YOLO_MODEL_NAME


def _find_project_root():
    script_dir = Path(__file__).resolve().parent
    # script_dir is collision_avoidance/monitor_vehicle/ and contains an `output`
    # sub-package; excluding it avoids matching the `output` marker against the
    # package dir itself.
    candidates = [script_dir.parent, script_dir.parent.parent]
    root_markers = (
        "dataset",
        "model",
        "output",
        "checkpoints",
        "_tmp_ml_depth_pro",
        "third_party",
    )

    for candidate in candidates:
        if any((candidate / marker).exists() for marker in root_markers):
            return candidate

    if script_dir.name.lower() == "main":
        return script_dir.parent
    return script_dir.parent.parent


def _prefer_existing_path(primary, *fallbacks):
    primary = Path(primary)
    for candidate in (primary, *fallbacks):
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return primary


def _video_path_from_name(video_name, dataset_dir=None):
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else Path(".")
    video_path = Path(video_name)
    stem = video_path.stem
    suffix = video_path.suffix or ".mp4"
    return _prefer_existing_path(
        dataset_dir / f"{stem}{suffix.lower()}",
        dataset_dir / f"{stem}{suffix.upper()}",
    )


def _model_path_from_name(model_name):
    model_path = Path(model_name)
    if model_path.is_absolute():
        return model_path

    return _prefer_existing_path(
        MODEL_WEIGHTS_DIR / model_path,
        MODEL_DIR / model_path,
        PROJECT_ROOT / model_path,
    )


PROJECT_ROOT = _find_project_root()
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_WEIGHTS_DIR = MODEL_DIR / "weights"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_CALIB_DIR = OUTPUT_DIR / "calibration"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
LEGACY_DEPTH_PRO_DIR = PROJECT_ROOT / "_tmp_ml_depth_pro"
DEPTH_PRO_SOURCE_DIR = THIRD_PARTY_DIR / "ml_depth_pro"

RUN_ID = f"{time.strftime('%Y%m%d_%H%M%S')}_test1_mvp"

YOLO_MODEL_PATH = _model_path_from_name(YOLO_MODEL_NAME)
DEPTH_PRO_REPO_SRC = _prefer_existing_path(
    DEPTH_PRO_SOURCE_DIR / "src",
    LEGACY_DEPTH_PRO_DIR / "src",
)
DEPTH_PRO_CHECKPOINT = _prefer_existing_path(
    CHECKPOINTS_DIR / "depth_pro.pt",
    LEGACY_DEPTH_PRO_DIR / "checkpoints/depth_pro.pt",
)

DEFAULT_VIDEO_PATH = _video_path_from_name(INPUT_VIDEO_NAME, DATASET_DIR)
DEFAULT_VIDEO_BASENAME = DEFAULT_VIDEO_PATH.stem if DEFAULT_VIDEO_PATH is not None else "input_video"
DEFAULT_OUTPUT_RUN_DIR = OUTPUT_DIR / DEFAULT_VIDEO_BASENAME
DEFAULT_OUTPUT_VIDEO = DEFAULT_OUTPUT_RUN_DIR / f"{DEFAULT_VIDEO_BASENAME}_mvp_annotated.mp4"
DEFAULT_TIMELINE_JSONL = DEFAULT_OUTPUT_RUN_DIR / f"{DEFAULT_VIDEO_BASENAME}_radar_timeline.jsonl"
DEFAULT_TIMELINE_PRETTY_JSON = DEFAULT_OUTPUT_RUN_DIR / f"{DEFAULT_VIDEO_BASENAME}_radar_timeline.pretty.json"
DEFAULT_CALIB_PATH = _prefer_existing_path(
    OUTPUT_CALIB_DIR / "depth_calibration.json",
    OUTPUT_DIR / "depth_calibration.json",
)
