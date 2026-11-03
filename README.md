# Conveyor & AMR Obstacle Detection

Video-based POC for detecting fallen components and generic debris in a conveyor/AMR operating area. The application uses YOLO detection and tracking, configurable floor/path regions, event generation, risk evaluation, and local evidence artifacts.

## Scope and important limits

- `Fallen_Component`: object observed on conveyor/table then falling and settling on the floor.
- `Unknown_Debris`: new generic obstacle appearing in the floor/path area without conveyor lineage.
- Existing floor objects are ignored during startup baseline.
- Demo calibration is explicitly marked placeholder. Real-world coordinates, dimensions, and metric distances are not reliable until calibrated using the actual camera.
- Red risk requires a confirmed obstacle, AMR-path overlap, AMR moving toward the obstacle, and decreasing distance over successive observations. The video-only demo cannot infer AMR identity/pose unless configured and detected.
- Each emitted event should save a frame, clip, and JSON payload; missing measurements remain null, never fabricated.

## Flow

1. Load video (file input; RTSP can be added through the same frame-source interface).
2. Read camera/calibration and polygon settings from `config/calibration.yaml`.
3. Run YOLO detections and persistent tracking.
4. Update object state: baseline, conveyor-associated, falling, landed/settling, or generic floor debris.
5. Validate candidate obstacle against floor and AMR-path polygons.
6. Assess risk with temporal approach evidence; uncertainty remains explicit.
7. Emit the agreed event payload and persist snapshot, clip, and JSON under `evidence/`.
8. Stream annotated frames and event log in Streamlit.

## Run

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Set `YOLO_MODEL` to a local compatible model path (default `yolo26n.pt`). Model availability depends on the installed Ultralytics release. Do not treat demo calibration as production calibration.
