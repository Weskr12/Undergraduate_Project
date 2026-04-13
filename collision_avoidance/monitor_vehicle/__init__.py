# -*- coding: utf-8 -*-
"""Monitor vehicle sub-package public API.

Importing this package does NOT trigger the pipeline — only the
backward-compat shim (collision_avoidance/monitorVehicle.py) does that.
"""

from .depth.calibration import calibration_template
from .output.payload import build_radar_payload, write_live_json
from .pipeline import run_mvp_pipeline

__all__ = [
    "run_mvp_pipeline",
    "calibration_template",
    "build_radar_payload",
    "write_live_json",
]
