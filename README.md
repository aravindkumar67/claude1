# Fallen Object Detection POC (v2.1 implementation baseline)

This repository separates the reusable computer-vision pipeline from the optional Streamlit demonstration UI.

## Important semantics
- `fallen_onto_ground` becomes true only after the tracked conveyor object leaves the configured conveyor ROI, satisfies the configured downward-drop threshold, and is confirmed stationary.
- Risk grading (low/medium/high) is performed **after** that boolean is set. A fall is not conditional on the yellow corridor grading polygons.
- `amr_approach` is an independent signal, intended to become true when a tracked AMR approaches a confirmed fallen object along the connected path. See implementation limitations below.
- Fixed yellow path polygons are provisional image-pixel geometry based on the reference clip view. They are not detected from video frames. Replace these values after actual camera/path calibration.
- Streamlit is optional. The core imports no Streamlit module.

## Setup in VM
1. Clone repository and enter it.
2. Create/activate a Python environment compatible with installed CUDA/PyTorch.
3. Install dependencies: `pip install -r requirements.txt`
4. Place your compatible YOLO26n weights at `models/yolo26n.pt`. The model weights are not committed.
5. Run UI: `streamlit run app.py`
6. Or run without Streamlit: `python run_pipeline.py --source path/to/video.mp4`
   - Camera: `python run_pipeline.py --source 0 --display`

## Configuration
- `config.yaml`: thresholds and provisional polygons.
- The initial calibration mode is image pixels. Pixel coordinates are not physical metres.
- Change model path, ROIs, and fixed corridor geometry to match your camera view.
- Do not treat the placeholder geometry as measured or validated real-world calibration.

## Event output
Events are written under `evidence/<signal_level>/<event_id>/` with snapshot, clip (when frames are available), and JSON payload. The payload includes:
- `fallen_onto_ground`: confirmed conveyor fall/landing.
- `amr_approach`: approach state (separate from fall and risk).
- `signal_level`: low, medium, high, or uncertain.
- `linked_to`: original fall event ID for escalation events.

## Validation status
This repository is a code baseline, not a claim of model performance. Run the video on your VM with your actual model weights and validate the measured detections, state transitions, event timing, geometry, AMR association, and evidence artifacts. The generic pretrained model may not identify your exact tote/container class. The configured path polygons are provisional.
