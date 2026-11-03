from collections import deque
from dataclasses import asdict
from pathlib import Path
import time, uuid, math
import cv2, numpy as np
from ultralytics import YOLO
from .geometry import bbox_footprint, grade_footprint, point_in_polygons
from .evidence import write_event

class FallenObjectPipeline:
    """Streamlit-free inference and event engine. Emits dict results per processed frame."""
    def __init__(self, config, source=0):
        self.cfg, self.source = config, source
        self.model = YOLO(config.model_path)
        self.histories = {}
        self.states = {}
        self.events = []
        self.buffer = deque(maxlen=max(1, int((config.evidence_before_s+config.evidence_after_s+1)*30)))
        self.running = False
        self._event_counter = 0

    def stop(self):
        self.running = False

    def _new_id(self, prefix="FALL"):
        self._event_counter += 1
        return f"{prefix}_{self._event_counter:04d}"

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_id = 0
        self.running = True
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = frame_id / fps
                self.buffer.append((ts, frame.copy()))
                result = self.model.track(frame, persist=True, tracker=self.cfg.tracker,
                    imgsz=self.cfg.imgsz, conf=self.cfg.confidence, iou=self.cfg.iou, verbose=False)[0]
                objects = []
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.int().cpu().numpy()
                    names = result.names
                    for box, tid, conf, cls in zip(boxes, ids, confs, classes):
                        obj = {"track_id":int(tid),"class_id":int(cls),"class_name":names[int(cls)],
                               "confidence":float(conf),"bbox":[float(x) for x in box]}
                        objects.append(obj)
                        self._update_track(obj, ts, frame, frame_id, fps)
                annotated = result.plot()
                yield {"frame_id":frame_id,"timestamp":ts,"frame":annotated,"objects":objects,
                       "events":list(self.events)}
                frame_id += 1
        finally:
            cap.release()
            self.running = False

    def _update_track(self, obj, ts, frame, frame_id, fps):
        tid, box = obj["track_id"], obj["bbox"]
        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        hist = self.histories.setdefault(tid, deque(maxlen=max(60,int(fps*4))))
        hist.append({"t":ts,"center":center,"box":box,"conf":obj["confidence"],"class":obj["class_name"],"frame":frame_id})
        state = self.states.setdefault(tid, {"state":"unknown","first_seen":ts,"event_created":False,"ground":False})
        # A conveyor candidate needs a sustained in-zone history before it can become a fall.
        roi = np.asarray(self.cfg.conveyor_roi, np.int32)
        in_conveyor = cv2.pointPolygonTest(roi, tuple(map(float,center)), False) >= 0
        if in_conveyor:
            if state["state"] in ("unknown","on_conveyor"):
                state["state"] = "on_conveyor"
                state["conveyor_since"] = state.get("conveyor_since",ts)
        elif state["state"] == "on_conveyor" and not state["event_created"]:
            state["state"] = "falling"
            state["fall_start"] = ts
            state["start_y"] = center[1]
        if state["state"] == "falling":
            elapsed = ts-state["fall_start"]
            drop = center[1]-state["start_y"]
            if elapsed <= self.cfg.drop_window_s and drop >= frame.shape[0]*self.cfg.drop_ratio:
                state["drop_confirmed"] = True
            recent = list(hist)[-max(2,int(fps*self.cfg.landing_duration_s)):]
            speed = 0.0
            if len(recent)>1:
                dt = max(recent[-1]["t"]-recent[0]["t"],1/fps)
                speed = math.dist(recent[-1]["center"],recent[0]["center"])/dt/frame.shape[0]
            if state.get("drop_confirmed") and len(recent)>=2 and speed <= self.cfg.landing_speed_ratio*fps:
                state["land_count"] = state.get("land_count",0)+1
            else:
                state["land_count"] = 0
            if state.get("land_count",0) >= max(2,int(fps*self.cfg.landing_duration_s)):
                state["state"]="landed"
                state["ground"]=True
                state["event_created"]=True
                self._emit_fall(obj, hist, frame, frame_id, ts, fps, state)
            elif elapsed > self.cfg.max_fall_s:
                state["state"]="uncertain"
                self._emit_uncertain(obj, frame, frame_id, ts, state)

    def _emit_fall(self, obj, hist, frame, frame_id, ts, fps, state):
        box = np.median(np.asarray([x["box"] for x in list(hist)[-5:]]),axis=0).tolist()
        footprint = bbox_footprint(box,self.cfg.footprint_ratio)
        corridors = [np.asarray(p,np.float32) for p in self.cfg.corridor_polygons]
        strips = [np.asarray(p,np.float32) for p in self.cfg.strip_polygons]
        level, interior, strip = grade_footprint(footprint,corridors,strips,self.cfg.high_overlap,self.cfg.medium_overlap)
        event = {"Event":"Obstacle","Anomaly_Type":"Fallen_Component","event_id":self._new_id(),
          "linked_to":None,"Coordinates":[float((box[0]+box[2])/2),float(box[3])],"Coordinates_Unit":"pixel",
          "Dimensions":[float(box[2]-box[0]),float(box[3]-box[1])],"Timestamp":ts,
          "Confidence_Score":obj["confidence"],"signal_level":level,"fallen_onto_ground":True,
          "amr_approach":False,"uncertain":False,"flags":[],"coords_source":"pixel",
          "pixel":[int((box[0]+box[2])/2),int(box[3])],"yolo_class":obj["class_name"],
          "track_id":obj["track_id"],"frame":frame_id,"amr_id":None,"amr_distance_m":None,
          "interior_overlap":interior,"strip_overlap":strip}
        self.events.append(event)
        clip = [f for t,f in self.buffer if ts-self.cfg.evidence_before_s <= t <= ts+self.cfg.evidence_after_s]
        # At landing, post-event frames are not available yet. Initial evidence is written with available frames.
        try: event["evidence"] = write_event(event,frame,clip,self.cfg.evidence_root,fps)
        except Exception as exc: event["flags"].append("evidence_failed"); event["evidence_error"]=str(exc)

    def _emit_uncertain(self,obj,frame,frame_id,ts,state):
        event={"Event":"Obstacle","Anomaly_Type":"Fallen_Component","event_id":self._new_id("UNCERTAIN"),
          "linked_to":None,"Coordinates":None,"Coordinates_Unit":"pixel","Dimensions":None,"Timestamp":ts,
          "Confidence_Score":obj["confidence"],"signal_level":"uncertain","fallen_onto_ground":False,
          "amr_approach":False,"uncertain":True,"flags":["fall_not_confirmed"],"coords_source":"pixel",
          "pixel":None,"yolo_class":obj["class_name"],"track_id":obj["track_id"],"frame":frame_id,
          "amr_id":None,"amr_distance_m":None,"interior_overlap":None,"strip_overlap":None}
        self.events.append(event)
