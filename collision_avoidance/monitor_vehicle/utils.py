# -*- coding: utf-8 -*-
"""Generic helpers with no local dependencies."""

import math
from pathlib import Path


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
