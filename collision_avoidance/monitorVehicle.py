# -*- coding: utf-8 -*-
"""
monitorVehicle.py

Thin shim kept for backward compatibility. The real implementation lives in
the `monitor_vehicle` sub-package. This file preserves the original
import-time behavior: when executed (or imported) it prints the constants
banner, warns about missing calibration, and runs smoke/full pipelines
according to the USER SETTINGS in monitor_vehicle/config.py.

Refer to monitor_vehicle/__init__.py for the public API.
"""

import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from monitor_vehicle import run_mvp_pipeline
from monitor_vehicle.config import (
    DEPTH_BACKEND,
    RUN_FULL_PIPELINE,
    RUN_SMOKE_TEST,
    SMOKE_TEST_MAX_FRAMES,
)
from monitor_vehicle.depth.calibration import calibration_template
from monitor_vehicle.output.payload import _validate_payload_schema
from monitor_vehicle.paths import (
    DEFAULT_CALIB_PATH,
    DEFAULT_LIVE_JSON,
    DEFAULT_OUTPUT_VIDEO,
    DEFAULT_TIMELINE_JSONL,
    DEFAULT_VIDEO_PATH,
)

print("MVP constants loaded")

if DEPTH_BACKEND != "depth_pro" and not DEFAULT_CALIB_PATH.exists():
    print("Calibration file not found:", DEFAULT_CALIB_PATH)
    print("Create it with 5 measured points, for example:")
    print(json.dumps(calibration_template(), ensure_ascii=False, indent=2))

if RUN_SMOKE_TEST:
    smoke_summary = run_mvp_pipeline(
        video_path=DEFAULT_VIDEO_PATH,
        output_video_path=DEFAULT_OUTPUT_VIDEO,
        timeline_jsonl_path=DEFAULT_TIMELINE_JSONL,
        live_json_path=DEFAULT_LIVE_JSON,
        calib_path=DEFAULT_CALIB_PATH,
        max_frames=SMOKE_TEST_MAX_FRAMES,
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
