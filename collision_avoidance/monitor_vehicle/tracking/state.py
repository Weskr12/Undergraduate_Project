# -*- coding: utf-8 -*-
"""Per-pipeline tracking state: history buffers, stale cleanup, stability checks."""

from collections import defaultdict, deque

import numpy as np

from ..config import (
    DEPTH_RECOVERY_MAX_CANDIDATES,
    DISTANCE_HISTORY_LEN,
    RADAR_TRAIL_LENGTH,
    TRACK_STABILITY_WINDOW,
    TRACK_STALE_FRAMES,
)
from ..detection.bbox import _bbox_bearing_deg
from ..risk import _is_stable_for_ttc


def _create_pipeline_tracking_state():
    return {
        "distance_history": defaultdict(lambda: deque(maxlen=DISTANCE_HISTORY_LEN)),
        "distance_confidence_history": defaultdict(lambda: deque(maxlen=DISTANCE_HISTORY_LEN)),
        "rejected_depth_candidates": defaultdict(lambda: deque(maxlen=DEPTH_RECOVERY_MAX_CANDIDATES)),
        "distance_band_state": {},
        "track_observation_history": defaultdict(lambda: deque(maxlen=TRACK_STABILITY_WINDOW)),
        "display_track_state": {},
        "track_hit_count": defaultdict(int),
        "last_seen_frame": {},
        "last_output_objects": {},
        "next_temp_track_id": -1,
        "last_depth_map": None,
        "radar_position_ema": {},
        "radar_position_history": defaultdict(lambda: deque(maxlen=RADAR_TRAIL_LENGTH)),
        "position_xz_history": defaultdict(lambda: deque(maxlen=DISTANCE_HISTORY_LEN)),
    }


def _update_track_stability(track_id, is_hook_turn, bbox, frame_w, tracking_state):
    bearing_deg = _bbox_bearing_deg(bbox, frame_w)
    center_x_norm = np.clip((0.5 * (bbox[0] + bbox[2])) / max(float(frame_w), 1.0), 0.0, 1.0)
    stable_for_ttc = False

    if track_id >= 0 and not is_hook_turn:
        track_observations = tracking_state["track_observation_history"][track_id]
        track_observations.append(
            {
                "center_x_norm": float(center_x_norm),
                "bearing_deg": float(bearing_deg),
            }
        )
        stable_for_ttc = _is_stable_for_ttc(track_observations)

    return bearing_deg, stable_for_ttc


def _cleanup_stale_tracks(frame_idx, tracking_state):
    stale_track_ids = [
        tid
        for tid, last_frame in tracking_state["last_seen_frame"].items()
        if (frame_idx - last_frame) > TRACK_STALE_FRAMES
    ]
    for tid in stale_track_ids:
        tracking_state["last_seen_frame"].pop(tid, None)
        tracking_state["last_output_objects"].pop(tid, None)
        tracking_state["distance_history"].pop(tid, None)
        tracking_state["distance_confidence_history"].pop(tid, None)
        tracking_state["rejected_depth_candidates"].pop(tid, None)
        tracking_state["distance_band_state"].pop(tid, None)
        tracking_state["track_observation_history"].pop(tid, None)
        tracking_state["display_track_state"].pop(tid, None)
        tracking_state["track_hit_count"].pop(tid, None)
        tracking_state["radar_position_ema"].pop(tid, None)
        tracking_state["radar_position_history"].pop(tid, None)
        tracking_state["position_xz_history"].pop(tid, None)
