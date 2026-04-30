# -*- coding: utf-8 -*-
"""YOLO detection + ByteTrack tracking wrapper."""

from ..config import CONF_THRESHOLDS, TRACKER_PATH, VEHICLE_CLASSES, YOLO_INFER_IMGSZ
from .bbox import _clip_bbox_xyxy


def _extract_detections_from_boxes(boxes, model_names, frame_shape, next_temp_track_id, force_temp_ids=False):
    frame_h, frame_w = frame_shape[:2]
    detections = []

    if boxes is None or len(boxes) == 0:
        return detections, next_temp_track_id

    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy().astype(float)

    track_ids = None
    if (not force_temp_ids) and getattr(boxes, "id", None) is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)

    for idx, bbox in enumerate(xyxy):
        class_id = int(clss[idx])
        conf = float(confs[idx])

        if conf < CONF_THRESHOLDS.get(class_id, 0.5):
            continue

        clipped_bbox = _clip_bbox_xyxy(bbox.tolist(), frame_w, frame_h)
        if clipped_bbox is None:
            continue

        if track_ids is None:
            track_id = next_temp_track_id
            next_temp_track_id -= 1
        else:
            track_id = int(track_ids[idx])

        class_name = model_names.get(class_id, str(class_id)) if isinstance(model_names, dict) else str(class_id)

        detections.append(
            {
                "track_id": int(track_id),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": clipped_bbox,
            }
        )

    return detections, next_temp_track_id


def _detect_with_tracking(frame_bgr, yolo_model, next_temp_track_id, runtime):
    track_exception = None
    infer_kwargs = {
        "classes": VEHICLE_CLASSES,
        "conf": 0.1,
        "device": runtime["device_str"],
        "imgsz": YOLO_INFER_IMGSZ,
        "verbose": False,
    }
    if runtime["use_half"]:
        infer_kwargs["half"] = True

    def _retry_without_half(fn, kwargs):
        try:
            return fn(**kwargs), None
        except Exception as exc:
            if not kwargs.get("half"):
                return None, exc

            fp32_kwargs = dict(kwargs)
            fp32_kwargs.pop("half", None)
            try:
                return fn(**fp32_kwargs), exc
            except Exception as retry_exc:
                return None, retry_exc

    try:
        track_call = lambda **kwargs: yolo_model.track(
            frame_bgr,
            persist=True,
            tracker=TRACKER_PATH,
            **kwargs,
        )
        track_results, track_exception = _retry_without_half(track_call, infer_kwargs)
    except Exception as exc:
        track_results, track_exception = None, exc

    if track_results and len(track_results) > 0:
        track_boxes = track_results[0].boxes
        if track_boxes is not None and len(track_boxes) > 0:
            has_track_ids = getattr(track_boxes, "id", None) is not None
            detections, next_temp_track_id = _extract_detections_from_boxes(
                track_boxes,
                yolo_model.names,
                frame_bgr.shape,
                next_temp_track_id,
                force_temp_ids=not has_track_ids,
            )
            mode_used = "track" if has_track_ids else "track_temp_ids"
            return detections, next_temp_track_id, mode_used, track_exception
        return [], next_temp_track_id, "track_empty", track_exception

    detect_call = lambda **kwargs: yolo_model(frame_bgr, **kwargs)
    detect_results, detect_exception = _retry_without_half(detect_call, infer_kwargs)
    if detect_exception is not None:
        track_exception = detect_exception
    if detect_results is None:
        return [], next_temp_track_id, "detect_failed", track_exception

    detect_boxes = detect_results[0].boxes if detect_results and len(detect_results) > 0 else None

    detections, next_temp_track_id = _extract_detections_from_boxes(
        detect_boxes,
        yolo_model.names,
        frame_bgr.shape,
        next_temp_track_id,
        force_temp_ids=True,
    )
    return detections, next_temp_track_id, "fallback_detect", track_exception
