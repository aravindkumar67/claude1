"""Optional Streamlit UI. Core pipeline modules do not import Streamlit."""
import streamlit as st
from core.config import load_config
from core.pipeline import FallenObjectPipeline

st.set_page_config(page_title="Fallen Object POC", layout="wide")
st.title("Fallen Object Detection & AMR Approach POC")
st.caption("UI wrapper only. Core pipeline is independently executable via run_pipeline.py.")

uploaded = st.file_uploader("Select reference video", type=["mp4","avi","mov","mkv"])
source_camera = st.checkbox("Use camera source instead")
config_path = st.text_input("Config file", "config.yaml")
if uploaded:
    st.video(uploaded)
start = st.button("Start processing", disabled=not (uploaded or source_camera))
if start:
    if source_camera:
        source = 0
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(uploaded.getbuffer())
            source = f.name
    cfg = load_config(config_path)
    pipeline = FallenObjectPipeline(cfg, source)
    frame_slot = st.empty()
    event_slot = st.empty()
    status = st.empty()
    progress = st.progress(0)
    for result in pipeline.run():
        frame_slot.image(result["frame"], channels="BGR", use_container_width=True)
        if result["events"]:
            event_slot.json(result["events"])
        status.write(f"Frame {result['frame_id']} | t={result['timestamp']:.2f}s | detections={len(result['objects'])}")
    st.success(f"Processing finished. {len(pipeline.events)} events recorded.")
    st.json(pipeline.events)
