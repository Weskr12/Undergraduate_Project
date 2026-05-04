# -*- coding: utf-8 -*-
"""Tests for pipeline-level track output stabilization helpers."""

from copy import deepcopy

from monitor_vehicle.pipeline import (
    _apply_detection_hold,
    _apply_track_class_vote,
    _bbox_iou_xyxy,
    _inherit_class_vote_from_held_overlap,
    _is_suppressed_by_active_overlap,
)
from monitor_vehicle.tracking.state import _create_pipeline_tracking_state


def _det(track_id=1, class_id=0, class_name="motorcycle", confidence=0.8, bbox=None):
    return {
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": bbox or [10, 10, 110, 110],
    }


def _obj(track_id=1, class_id=0, class_name="motorcycle", status="active", bbox=None):
    return {
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": 0.8,
        "bbox_xyxy": bbox or [10, 10, 110, 110],
        "track_status": status,
        "hook_turn_detected": False,
        "distance_m": 7.0,
    }


class TestBboxIou:
    def test_identical_boxes(self):
        assert _bbox_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_non_overlapping_boxes(self):
        assert _bbox_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


class TestTrackClassVote:
    def test_majority_class_wins_for_track(self):
        state = _create_pipeline_tracking_state()

        first = _apply_track_class_vote(_det(class_id=1, class_name="car"), state)
        second = _apply_track_class_vote(_det(class_id=0, class_name="motorcycle"), state)
        third = _apply_track_class_vote(_det(class_id=0, class_name="motorcycle"), state)

        assert first["class_name"] == "car"
        assert second["class_name"] == "car"
        assert third["class_id"] == 0
        assert third["class_name"] == "motorcycle"

    def test_temporary_track_bypasses_vote(self):
        state = _create_pipeline_tracking_state()
        det = _det(track_id=-1, class_id=1, class_name="car")

        assert _apply_track_class_vote(det, state) == det

    def test_confidence_weighted_class_wins(self):
        state = _create_pipeline_tracking_state()

        _apply_track_class_vote(_det(class_id=1, class_name="car", confidence=0.2), state)
        _apply_track_class_vote(_det(class_id=1, class_name="car", confidence=0.2), state)
        voted = _apply_track_class_vote(_det(class_id=0, class_name="motorcycle", confidence=0.8), state)

        assert voted["class_id"] == 0
        assert voted["class_name"] == "motorcycle"

    def test_new_track_inherits_vote_history_from_overlapping_held_track(self):
        state = _create_pipeline_tracking_state()
        state["last_seen_frame"][1] = 9
        state["last_output_objects"][1] = _obj(track_id=1, class_id=0, class_name="motorcycle")
        state["class_vote_history"][1].append((0, "motorcycle", 0.8))
        state["class_vote_history"][1].append((0, "motorcycle", 0.7))
        state["class_vote_state"][1] = (0, "motorcycle")

        new_det = _det(track_id=2, class_id=1, class_name="car", confidence=0.6, bbox=[12, 12, 112, 112])
        _inherit_class_vote_from_held_overlap(new_det, frame_idx=10, tracking_state=state)
        voted = _apply_track_class_vote(new_det, state)

        assert voted["class_id"] == 0
        assert voted["class_name"] == "motorcycle"


class TestHeldOverlapSuppression:
    def test_same_class_high_overlap_suppresses_held(self):
        held = _obj(status="held")
        active = [_obj(track_id=2, status="active", bbox=[12, 12, 112, 112])]

        assert _is_suppressed_by_active_overlap(held, active, iou_threshold=0.5) is True

    def test_different_class_does_not_suppress_held(self):
        held = _obj(status="held", class_id=0, class_name="motorcycle")
        active = [_obj(track_id=2, class_id=1, class_name="car", bbox=[12, 12, 112, 112])]

        assert _is_suppressed_by_active_overlap(held, active, iou_threshold=0.5) is False

    def test_apply_detection_hold_skips_duplicate_held_object(self):
        state = _create_pipeline_tracking_state()
        previous = _obj(track_id=1, status="active")
        state["last_seen_frame"][1] = 9
        state["last_output_objects"][1] = deepcopy(previous)

        active_objects = [_obj(track_id=2, status="active", bbox=[12, 12, 112, 112])]
        result = _apply_detection_hold(None, 10, active_objects, state)

        assert result == active_objects

    def test_apply_detection_hold_keeps_non_overlapping_held_object(self):
        state = _create_pipeline_tracking_state()
        previous = _obj(track_id=1, status="active")
        state["last_seen_frame"][1] = 9
        state["last_output_objects"][1] = deepcopy(previous)

        active_objects = [_obj(track_id=2, status="active", bbox=[300, 300, 400, 400])]
        result = _apply_detection_hold(None, 10, active_objects, state)

        assert len(result) == 2
        assert result[1]["track_status"] == "held"
        assert result[1]["class_name"] == previous["class_name"]
        assert result[1]["distance_m"] == previous["distance_m"]
