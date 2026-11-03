import json, os, cv2, tempfile
from pathlib import Path

def write_event(event, snapshot, frames, root="evidence", fps=24.0):
    level = event.get("signal_level", "uncertain")
    folder = Path(root) / level / f"{event['event_id']}"
    folder.mkdir(parents=True, exist_ok=True)
    snap_path, clip_path, json_path = folder/"snapshot.jpg", folder/"clip.mp4", folder/"data.json"
    ok, encoded = cv2.imencode(".jpg", snapshot)
    if not ok:
        event.setdefault("flags", []).append("evidence_failed")
        raise IOError("Could not encode event snapshot")
    tmp_snap = snap_path.with_suffix(".jpg.tmp")
    tmp_snap.write_bytes(encoded.tobytes()); os.replace(tmp_snap, snap_path)
    if frames:
        h, w = frames[0].shape[:2]
        tmp_clip = folder/"clip.tmp.mp4"
        writer = cv2.VideoWriter(str(tmp_clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
        try:
            for frame in frames:
                if frame.shape[:2] != (h,w):
                    frame = cv2.resize(frame,(w,h))
                writer.write(frame)
        finally:
            writer.release()
        if tmp_clip.exists():
            os.replace(tmp_clip, clip_path)
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_json, json_path)
    return {"snapshot":str(snap_path),"clip":str(clip_path) if clip_path.exists() else None,"json":str(json_path)}
