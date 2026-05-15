# -*- coding: utf-8 -*-
"""Right-side radar panel rendering + final output frame composition."""

import math
from collections import defaultdict, deque

import cv2
import numpy as np

from ..config import (
    HFOV_DEG,
    OUTPUT_MAX_HEIGHT,
    OUTPUT_MAX_WIDTH,
    RADAR_BANDS,
    RADAR_BAND_LABELS,
    RADAR_BEARING_CUTOFF_DEG,
    RADAR_BG_COLOR,
    RADAR_BORDER_COLOR,
    RADAR_MAX_DISTANCE_M,
    RADAR_OVERLAP_MARGIN_PX,
    RADAR_OVERLAP_STEP_PX,
    RADAR_PANEL_WIDTH,
    RADAR_PLOT_BG_COLOR,
    RADAR_POSITION_EMA_ALPHA,
    RADAR_SHOW_TEMP_TRACKS,
    RADAR_SUBTEXT_COLOR,
    RADAR_TEXT_COLOR,
    RADAR_TRAIL_FADE_MIN_ALPHA,
    RADAR_TRAIL_LENGTH,
    RADAR_TRAIL_MIN_SEGMENTS,
    RADAR_TRAIL_THICKNESS,
    RISK_COLORS,
    RISK_SEVERITY,
)
from ..utils import to_float_or_none


def _empty_radar_state():
    return {
        "ema": {},
        "history": defaultdict(lambda: deque(maxlen=RADAR_TRAIL_LENGTH)),
    }


def _smooth_base_radar_position(track_id, base_position, ema_state):
    if base_position is None:
        return None
    base_x, base_y = float(base_position[0]), float(base_position[1])
    if track_id < 0:
        return (int(round(base_x)), int(round(base_y)))

    prev = ema_state.get(track_id)
    if prev is None:
        smooth_x, smooth_y = base_x, base_y
    else:
        prev_x, prev_y = prev
        smooth_x = (1.0 - RADAR_POSITION_EMA_ALPHA) * prev_x + RADAR_POSITION_EMA_ALPHA * base_x
        smooth_y = (1.0 - RADAR_POSITION_EMA_ALPHA) * prev_y + RADAR_POSITION_EMA_ALPHA * base_y

    ema_state[track_id] = (smooth_x, smooth_y)
    return (int(round(smooth_x)), int(round(smooth_y)))


def _append_radar_trail(track_id, placed_position, history_state):
    if placed_position is None or track_id < 0:
        return
    deque_for_track = history_state.get(track_id)
    if deque_for_track is None:
        deque_for_track = deque(maxlen=RADAR_TRAIL_LENGTH)
        history_state[track_id] = deque_for_track
    deque_for_track.append((int(placed_position[0]), int(placed_position[1])))


def _draw_radar_trails(panel, radar_detections, history_state, plot_left, plot_top, plot_right, plot_bottom):
    if not history_state:
        return

    for det in radar_detections:
        track_id = int(det.get("track_id", -1))
        if track_id < 0:
            continue
        points = history_state.get(track_id)
        if points is None or len(points) < RADAR_TRAIL_MIN_SEGMENTS:
            continue

        color = RISK_COLORS.get(det.get("risk_level"), RISK_COLORS["unknown"])
        point_list = list(points)
        segment_count = len(point_list) - 1
        for seg_idx in range(segment_count):
            p0 = point_list[seg_idx]
            p1 = point_list[seg_idx + 1]
            if not _segment_inside_plot(p0, p1, plot_left, plot_top, plot_right, plot_bottom):
                continue
            alpha_progress = (seg_idx + 1) / float(segment_count)
            alpha = RADAR_TRAIL_FADE_MIN_ALPHA + alpha_progress * (1.0 - RADAR_TRAIL_FADE_MIN_ALPHA)
            blended = _blend_against_plot_bg(color, alpha)
            cv2.line(panel, p0, p1, blended, RADAR_TRAIL_THICKNESS, lineType=cv2.LINE_AA)


def _segment_inside_plot(p0, p1, plot_left, plot_top, plot_right, plot_bottom):
    x0, y0 = p0
    x1, y1 = p1
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)
    if max_x < plot_left or min_x > plot_right:
        return False
    if max_y < plot_top or min_y > plot_bottom:
        return False
    return True


def _blend_against_plot_bg(color, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = tuple(
        int(round(alpha * float(color[i]) + (1.0 - alpha) * float(RADAR_PLOT_BG_COLOR[i])))
        for i in range(3)
    )
    return blended


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


def _display_distance_m(det):
    distance = det.get("display_distance_m")
    if distance is None:
        distance = det.get("distance_m")
    return to_float_or_none(distance)


def _iter_radar_detections(detections):
    radar_detections = []

    for det in detections:
        if det.get("hook_turn_detected"):
            continue
        if int(det.get("track_id", -1)) < 0 and not RADAR_SHOW_TEMP_TRACKS:
            continue
        distance_m = _display_distance_m(det)
        bearing_deg = to_float_or_none(det.get("bearing_deg"))
        if distance_m is None or bearing_deg is None:
            continue
        radar_detections.append(det)

    return radar_detections


def _distance_to_plot_fraction(distance_m):
    distance = max(0.0, to_float_or_none(distance_m) or 0.0)

    if distance <= 8.0:
        progress = np.clip(distance / 8.0, 0.0, 1.0)
        return float(1.0 - progress / 3.0)
    if distance <= 15.0:
        progress = np.clip((distance - 8.0) / 7.0, 0.0, 1.0)
        return float((2.0 / 3.0) - progress / 3.0)
    progress = np.clip((distance - 15.0) / max(RADAR_MAX_DISTANCE_M - 15.0, 1e-6), 0.0, 1.0)
    return float((1.0 / 3.0) - progress / 3.0)


def _radar_target_position(det, plot_left, plot_top, plot_right, plot_bottom):
    distance_m = _display_distance_m(det)
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
    distance_m = _display_distance_m(det) or RADAR_MAX_DISTANCE_M
    if risk_level in {"critical", "high"}:
        return 10
    if distance_m <= 8.0:
        return 9
    return 7


def _select_radar_list_targets(radar_detections, max_items=6):
    return sorted(
        radar_detections,
        key=lambda det: (
            _display_distance_m(det) if _display_distance_m(det) is not None else float("inf"),
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
    center_x = plot_left + plot_width // 2

    cv2.line(panel, (center_x, plot_top), (center_x, plot_bottom), RADAR_BORDER_COLOR, 1)
    band_count = max(1, len(RADAR_BANDS))
    for idx in range(1, band_count):
        band_y = plot_top + int(round(plot_height * idx / float(band_count)))
        cv2.line(panel, (plot_left, band_y), (plot_right, band_y), RADAR_BORDER_COLOR, 1)

    cv2.putText(panel, "L", (plot_left + 2, plot_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_SUBTEXT_COLOR, 1)
    cv2.putText(panel, "C", (center_x - 6, plot_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_SUBTEXT_COLOR, 1)
    cv2.putText(panel, "R", (plot_right - 12, plot_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_SUBTEXT_COLOR, 1)

    for band_idx, band in enumerate(RADAR_BANDS):
        band_top = plot_top + int(round(plot_height * band_idx / float(band_count)))
        cv2.putText(
            panel,
            RADAR_BAND_LABELS[band],
            (plot_left + 8, band_top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 if len(RADAR_BAND_LABELS[band]) > 6 else 0.48,
            RADAR_SUBTEXT_COLOR,
            1,
        )


def _draw_radar_targets(panel, radar_detections, plot_left, plot_top, plot_right, plot_bottom, radar_state):
    sorted_targets = sorted(
        radar_detections,
        key=lambda det: (
            -RISK_SEVERITY.get(det.get("risk_level", "unknown"), 0),
            _display_distance_m(det) if _display_distance_m(det) is not None else float("inf"),
            abs(to_float_or_none(det.get("bearing_deg")) or 0.0),
        ),
    )
    placed_targets = []
    ema_state = radar_state["ema"]
    history_state = radar_state["history"]

    for det in sorted_targets:
        track_id = int(det.get("track_id", -1))
        raw_base = _radar_target_position(det, plot_left, plot_top, plot_right, plot_bottom)
        base_position = _smooth_base_radar_position(track_id, raw_base, ema_state)
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
        _append_radar_trail(track_id, (x, y), history_state)

        distance_m = _display_distance_m(det) or 0.0
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
        distance_m = _display_distance_m(det) or 0.0
        bearing_label = _bearing_label(det.get("bearing_deg"))
        text = f"{bearing_label} {_short_class_name(det['class_name'])} {distance_m:>4.1f}m"
        cv2.circle(panel, (20, row_y - 5), 5, color, -1)
        cv2.putText(panel, text, (32, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_TEXT_COLOR, 1)
        row_y += 20

    if not list_targets:
        cv2.putText(panel, "no valid targets", (16, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RADAR_SUBTEXT_COLOR, 1)


def _draw_radar_panel(output_frame, detections, radar_state):
    frame_h, frame_w = output_frame.shape[:2]
    panel_x0 = frame_w - RADAR_PANEL_WIDTH
    panel = output_frame[:, panel_x0:frame_w]
    panel[:] = RADAR_BG_COLOR

    radar_detections = _iter_radar_detections(detections)
    min_distance = min((_display_distance_m(det) for det in radar_detections), default=None)

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
    _draw_radar_trails(
        panel,
        radar_detections,
        radar_state["history"],
        plot_left,
        plot_top,
        plot_right,
        plot_bottom,
    )
    _draw_radar_targets(panel, radar_detections, plot_left, plot_top, plot_right, plot_bottom, radar_state)
    _draw_radar_list(panel, radar_detections, frame_h)


def _compose_output_frame(frame_bgr, detections, radar_state=None):
    frame_h, frame_w = frame_bgr.shape[:2]
    canvas = np.zeros((frame_h, frame_w + RADAR_PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :frame_w] = frame_bgr
    if radar_state is None:
        radar_state = _empty_radar_state()
    _draw_radar_panel(canvas, detections, radar_state)
    return canvas


def _compute_output_video_size(frame_w, frame_h):
    output_w = int(frame_w + RADAR_PANEL_WIDTH)
    output_h = int(frame_h)
    scale_w = OUTPUT_MAX_WIDTH / max(float(output_w), 1.0)
    scale_h = OUTPUT_MAX_HEIGHT / max(float(output_h), 1.0)
    scale = min(1.0, scale_w, scale_h)
    resized_w = max(2, int(round(output_w * scale)))
    resized_h = max(2, int(round(output_h * scale)))

    if resized_w % 2 != 0:
        resized_w -= 1
    if resized_h % 2 != 0:
        resized_h -= 1
    return resized_w, resized_h


def _resize_output_frame(output_frame, target_size):
    target_w, target_h = target_size
    current_h, current_w = output_frame.shape[:2]
    if current_w == target_w and current_h == target_h:
        return output_frame
    return cv2.resize(output_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
