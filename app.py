"""Streamlit UI and video pipeline for the obstacle detection POC."""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "calibration.yaml"


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.int32), tuple(map(float, point)), False) >= 0


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def event_payload(anomaly_type, event_id, box, timestamp, confidence, track_id, frame_no,
                  floor_hit, path_hit, uncertain, risk, coords_source="pixel"):
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = center(box)
    return {
        "Event": "Obstacle", "Anomaly_Type": anomaly_type, "event_id": event_id,
        "linked_to": None, "Coordinates": None, "Coordinates_Unit": None,
        "Dimensions": None, "Timestamp": timestamp, "Confidence_Score": float(confidence),
        "signal_level": risk, "fallen_onto_ground": bool(floor_hit),
        "amr_approach": None, "uncertain": bool(uncertain), "flags": [],
        "coords_source": coords_source, "pixel": {"x": float(cx), "y": float(cy), "box": [x1,y1,x2,y2]},
        "yolo_class": None, "track_id": track_id, "frame": int(frame_no),
        "amr_id": None, "amr_distance_m": None, "interior_overlap": None,
        "strip_overlap": bool(path_hit)
    }


def save_evidence(cfg, event, frame, clip_frames):
    root = ROOT / cfg["evidence"]["root"]
    folder = root / str(event["signal_level"]) / f'{event["event_id"]}_{datetime.now().strftime("%Y%m%dT%H%M%S")}'
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / "snapshot.jpg"
    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f"Could not write snapshot: {image_path}")
    json_path = folder / "event.json"
    json_path.write_text(json.dumps(event, indent=2), encoding="utf-8")
    clip_path = folder / "clip.mp4"
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    try:
        for image in clip_frames:
            if image.shape[:2] == (h, w):
                writer.write(image)
    finally:
        writer.release()
    return str(folder)


def run_video(video_path, cfg, model, display, stop_flag):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error("Video could not be opened.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    baseline_frames = max(1, int(cfg["tracking"]["baseline_seconds"] * fps))
    stationary_frames = max(1, int(cfg["tracking"]["stationary_seconds"] * fps))
    cooldown_frames = max(1, int(cfg["tracking"]["event_cooldown_seconds"] * fps))
    floor_poly, path_poly = cfg["regions"]["floor_polygon"], cfg["regions"]["amr_path_polygon"]
    states, trail, last_event, clipbuf = {}, defaultdict(lambda: deque(maxlen=int(fps * cfg["evidence"]["clip_seconds_before"]))), {}, deque(maxlen=int(fps * cfg["evidence"]["clip_seconds_before"]))
    frame_no, baseline = 0, {}
    events = []
    event_panel = st.container()
    while cap.isOpened() and not stop_flag():
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        result = model.track(frame, persist=True, conf=cfg["tracking"]["confidence"], iou=cfg["tracking"]["iou"], tracker=cfg["tracking"]["tracker"], verbose=False)[0]
        annotated = result.plot()
        ids = result.boxes.id.int().cpu().tolist() if result.boxes is not None and result.boxes.id is not None else []
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0,4))
        classes = result.boxes.cls.int().cpu().tolist() if result.boxes is not None else []
        confs = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        current = {}
        for tid, box, cls, conf in zip(ids, boxes, classes, confs):
            c = center(box)
            current[tid] = (box, cls, conf, c)
            trail[tid].append((frame_no, c[0], c[1]))
            if frame_no <= baseline_frames:
                baseline[tid] = (c, point_in_polygon(c, floor_poly))
        if frame_no > baseline_frames:
            for tid, (box, cls, conf, c) in current.items():
                in_floor = point_in_polygon(c, floor_poly)
                in_path = point_in_polygon(c, path_poly)
                history = trail[tid]
                # Vertical displacement and a stable floor position indicate candidate fall/settling.
                dy = history[-1][2] - history[0][2] if len(history) > 1 else 0
                tail = list(history)[-stationary_frames:]
                settled = len(tail) >= stationary_frames and max(p[2] for p in tail) - min(p[2] for p in tail) < 8 and max(p[1] for p in tail) - min(p[1] for p in tail) < 8
                initial_floor = baseline.get(tid, (None, False))[1]
                anomaly = "Fallen_Component" if dy >= cfg["tracking"]["fall_min_vertical_pixels"] else "Unknown_Debris"
                # Existing objects seen on floor in startup baseline are not reported.
                if initial_floor and in_floor:
                    continue
                if in_floor and settled and in_path and frame_no - last_event.get(tid, -cooldown_frames) >= cooldown_frames:
                    risk = "uncertain" if cfg["risk"]["demo_mode"] else "medium"
                    uncertain = True
                    event = event_payload(anomaly, str(uuid.uuid4()), box, iso_now(), conf, tid, frame_no, True, in_path, uncertain, risk)
                    event["yolo_class"] = model.names.get(cls, str(cls))
                    event["flags"] = ["DEMO_CALIBRATION", "AMR_POSE_NOT_CONFIGURED"]
                    clip_frames = list(clipbuf) + [frame.copy()]
                    try:
                        event["evidence_path"] = save_evidence(cfg, event, annotated, clip_frames)
                    except Exception as exc:
                        event["flags"].append("EVIDENCE_SAVE_FAILED")
                        event["evidence_error"] = str(exc)
                        st.error(f"Evidence write failed: {exc}")
                    events.append(event)
                    last_event[tid] = frame_no
        clipbuf.append(frame.copy())
        display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        with event_panel:
            st.caption(f"Frames processed: {frame_no} | Events: {len(events)}")
            if events:
                st.dataframe([{k: e.get(k) for k in ("event_id", "Anomaly_Type", "signal_level", "track_id", "Timestamp", "evidence_path")} for e in events], use_container_width=True)
    cap.release()
    st.success(f"Finished. Processed {frame_no} frames; emitted {len(events)} event(s).")
    if events:
        st.download_button("Download event JSON", json.dumps(events, indent=2), file_name="events.json", mime="application/json")


st.set_page_config(page_title="AMR Obstacle Detection", layout="wide")
st.title("Conveyor & AMR Obstacle Detection")
cfg = load_config()
if cfg["calibration"]["status"] == "DEMO_PLACEHOLDER":
    st.warning("DEMO / PLACEHOLDER CALIBRATION: metric coordinates, dimensions, and AMR distances are not valid until real-camera calibration is entered.")
model_path = os.getenv("YOLO_MODEL", cfg["tracking"]["model"])
upload = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
start = st.button("Run detection", type="primary", disabled=upload is None)
if start and upload:
    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getvalue())
        video_path = tmp.name
    try:
        with st.spinner(f"Loading model: {model_path}"):
            model = YOLO(model_path)
        frame_display = st.empty()
        run_video(video_path, cfg, model, frame_display, lambda: False)
    except Exception as exc:
        st.exception(exc)
    finally:
        Path(video_path).unlink(missing_ok=True)
