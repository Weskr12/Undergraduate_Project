# -*- coding: utf-8 -*-
"""Draw per-detection bboxes/labels and frame summary text onto the frame."""

import cv2

from ..config import (
    ANNOTATION_BOX_OUTLINE_THICKNESS,
    ANNOTATION_BOX_THICKNESS,
    ANNOTATION_FONT_SCALE,
    ANNOTATION_FONT_THICKNESS,
    ANNOTATION_LABEL_BG_COLOR,
    ANNOTATION_LABEL_TEXT_COLOR,
    ANNOTATION_TEXT_PADDING_X,
    ANNOTATION_TEXT_PADDING_Y,
    RISK_COLORS,
)


def _draw_detection(frame_bgr, det):
    x1, y1, x2, y2 = det["bbox_xyxy"]
    risk = det["risk_level"]
    color = RISK_COLORS.get(risk, RISK_COLORS["unknown"])

    if det["hook_turn_detected"]:
        color = RISK_COLORS["info"]

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), ANNOTATION_BOX_OUTLINE_THICKNESS)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, ANNOTATION_BOX_THICKNESS)

    dist_txt = "n/a" if det["distance_m"] is None else f"{det['distance_m']:.1f}m"
    ttc_txt = "n/a" if det.get("ttc_s") is None else f"{det['ttc_s']:.1f}s"
    label = (
        f"ID:{det['track_id']} {det['class_name']} "
        f"{dist_txt} ttc:{ttc_txt} {det['risk_level']} b:{det['bearing_deg']:.1f}"
    )
    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        ANNOTATION_FONT_SCALE,
        ANNOTATION_FONT_THICKNESS,
    )
    label_x1 = x1
    label_y2 = max(text_h + baseline + ANNOTATION_TEXT_PADDING_Y * 2, y1 - 6)
    label_y1 = max(0, label_y2 - text_h - baseline - ANNOTATION_TEXT_PADDING_Y * 2)
    label_x2 = min(frame_bgr.shape[1] - 1, label_x1 + text_w + ANNOTATION_TEXT_PADDING_X * 2)
    cv2.rectangle(
        frame_bgr,
        (label_x1, label_y1),
        (label_x2, label_y2),
        ANNOTATION_LABEL_BG_COLOR,
        -1,
    )
    cv2.rectangle(
        frame_bgr,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color,
        2,
    )
    text_origin = (
        label_x1 + ANNOTATION_TEXT_PADDING_X,
        label_y2 - baseline - ANNOTATION_TEXT_PADDING_Y,
    )
    cv2.putText(
        frame_bgr,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        ANNOTATION_FONT_SCALE,
        ANNOTATION_LABEL_TEXT_COLOR,
        ANNOTATION_FONT_THICKNESS,
    )
    cv2.putText(
        frame_bgr,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        ANNOTATION_FONT_SCALE,
        color,
        1,
    )


def _draw_frame_summary(frame, frame_idx, payload):
    min_distance = payload["summary"]["min_distance_m"]
    min_ttc_s = payload["summary"]["min_ttc_s"]
    min_distance_txt = "n/a" if min_distance is None else f"{min_distance:.1f}m"
    min_ttc_txt = "n/a" if min_ttc_s is None else f"{min_ttc_s:.1f}s"
    info_text = (
        f"Frame:{frame_idx} Objects:{payload['summary']['object_count']} "
        f"MinDist:{min_distance_txt} MinTTC:{min_ttc_txt}"
    )
    cv2.putText(frame, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
