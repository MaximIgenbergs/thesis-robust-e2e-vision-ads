from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json
import numpy as np
from PIL import Image


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


class DataRunLogger:
    """
    Lightweight run logger for data collection runs.
    - Creates <run_dir>/manifest.json
    """

    def __init__(
        self,
        run_dir: Path,
        map_name: str,
        source: str,
        sim_app: Optional[Path | str] = None,
        data_dir: Optional[Path | str] = None,
        raw_pd_dir: Optional[Path | str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.manifest_path = self.run_dir / "manifest.json"
        run_id = self.run_dir.name
        self.state: Dict[str, Any] = {
            "schema_version": "dc-1.0",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "map_name": map_name,
            "source": source,
            "paths": {
                "sim_app": (str(sim_app) if sim_app else None),
                "data_dir": (str(data_dir) if data_dir else str(self.run_dir)),
                "raw_pd_dir": (str(raw_pd_dir) if raw_pd_dir else None),
            },
            "frames_written": 0,
            "dropped_frames": 0,
            "extras": extras or {},
        }
        _dump_json(self.manifest_path, self.state)

    def add_frames(self, n: int = 1) -> None:
        self.state["frames_written"] = int(self.state.get("frames_written", 0)) + int(n)
        _dump_json(self.manifest_path, self.state)

    def add_dropped(self, n: int = 1) -> None:
        self.state["dropped_frames"] = int(self.state.get("dropped_frames", 0)) + int(n)
        _dump_json(self.manifest_path, self.state)

    def set_extra(self, key: str, value: Any) -> None:
        self.state.setdefault("extras", {})[key] = value
        _dump_json(self.manifest_path, self.state)


# ---- Helpers ----


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_run_dir(base_dir: Path, prefix: str) -> Path:
    """
    Creates <base_dir>/<prefix>_YYYYMMDD-HHMMSS and returns it.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (base_dir / f"{prefix}_{ts}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_image(img_np: np.ndarray, path: Path) -> None:
    """
    Saves an RGB image (uint8). Grayscale gets stacked to 3 channels.
    """
    arr = img_np
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    Image.fromarray(arr).save(str(path), format="JPEG", quality=95)


def write_frame_record(path: Path, steer: float, throttle: float, track_id: int, frame_idx_in_run: int) -> None:
    rec: Dict[str, Any] = {
        "user/angle": f"{float(steer)}",
        "user/throttle": f"{float(throttle)}",
        "meta/track_id": int(track_id),
        "meta/frame": int(frame_idx_in_run),
    }
    path.write_text(json.dumps(rec, indent=2))
