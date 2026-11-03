from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class AppConfig:
    model_path: str = "models/yolo26n.pt"
    imgsz: int = 960
    confidence: float = 0.25
    iou: float = 0.50
    tracker: str = "bytetrack.yaml"
    conveyor_roi: tuple = ((0, 250), (500, 250), (650, 500), (0, 520))
    corridor_polygons: tuple = (((450, 380), (1280, 180), (1280, 380), (650, 620)),)
    strip_polygons: tuple = (((450, 380), (1280, 180), (1280, 195), (465, 395)),)
    conveyor_dwell_s: float = 0.4
    drop_ratio: float = 0.05
    drop_window_s: float = 1.0
    landing_speed_ratio: float = 0.003
    landing_duration_s: float = 0.25
    max_fall_s: float = 2.0
    footprint_ratio: float = 0.35
    high_overlap: float = 0.01
    medium_overlap: float = 0.01
    amr_min_conf: float = 0.4
    amr_min_track_frames: int = 10
    approach_window_s: float = 0.5
    approach_fraction: float = 0.8
    approach_net_ratio: float = 0.03
    amr_speed_ratio: float = 0.002
    heading_tolerance_deg: float = 45.0
    red_confirm_frames: int = 3
    red_clear_s: float = 0.25
    evidence_before_s: float = 1.5
    evidence_after_s: float = 2.0
    evidence_root: str = "evidence"

def load_config(path="config.yaml"):
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a mapping")
    cfg = AppConfig(**{k: v for k, v in raw.items() if k in AppConfig.__dataclass_fields__})
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"YOLO weights not found: {cfg.model_path}")
    if not 0 < cfg.confidence < 1 or not 0 < cfg.iou < 1:
        raise ValueError("confidence and iou must be between 0 and 1")
    return cfg
