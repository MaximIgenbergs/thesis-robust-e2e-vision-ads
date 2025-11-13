from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import numpy as np
from PIL import Image

def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))

class DataRunLogger:
    """
    Lightweight run logger for data collection runs.
    - Creates <run_dir>/manifest.json
    """

    def __init__(self, run_dir: Path, map_name: str, source: str,
                 sim_app: Optional[Path | str] = None,
                 data_dir: Optional[Path | str] = None,
                 raw_pd_dir: Optional[Path | str] = None,
                 extras: Optional[Dict[str, Any]] = None):
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
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
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
        arr = np.stack([arr]*3, axis=-1)
    Image.fromarray(arr).save(str(path), format="JPEG", quality=95)

def write_frame_record(path: Path, steer: float, throttle: float, track_id: int, topo_id: int, frame_idx_in_run: int) -> None:
    rec: Dict[str, Any] = {
        "user/angle": f"{float(steer)}",
        "user/throttle": f"{float(throttle)}",
        "meta/track_id": int(track_id),
        "meta/topo_id":  int(topo_id),
        "meta/frame":    int(frame_idx_in_run),
    }
    path.write_text(json.dumps(rec, indent=2))

# ---- Data Format Conversion from ScenarioOutcomeWriter in PerturbationDrive ----

_TOPO_PATTERNS = [
    re.compile(r"(?:track|road|map|generated)[-_]?(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)$"),
]

def _infer_topo_id_from_filename(path: Path) -> Optional[int]:
    stem = path.stem
    for pat in _TOPO_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def _coerce_action(raw) -> Tuple[float, float]:
    """Accepts [s,t] or [[s,t], ...]"""
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return float(raw[0]), float(raw[1])

def convert_outputs(pd_logs_dir: Path, out_pairs_dir: Path) -> None:
    """
    Converts PerturbationDrive ScenarioOutcomeWriter outputs into the required data format:
    out_pairs_dir/
      image_000001.jpg
      record_000001.json
      ...
    """
    out_pairs_dir.mkdir(parents=True, exist_ok=True)

    json_files: List[Path] = sorted([p for p in pd_logs_dir.iterdir() if p.suffix.lower() == ".json"])
    if not json_files:
        print(f"[Data Conversion] no JSON logs in {pd_logs_dir}")
        return

    start_idx = 1
    topo_counter = 0
    run_uid_counter = 0

    for jf in json_files:
        base = jf.with_suffix("")
        image_dir = Path(str(base) + "___0_original")
        if not image_dir.exists():
            print(f"[Data Conversion] image folder not found for {jf.name}: {image_dir}")
            continue

        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError:
            print(f"[Data Conversion] json decode error: {jf}")
            continue

        entries = [data] if isinstance(data, dict) else list(data)
        topo_id = _infer_topo_id_from_filename(jf)
        if topo_id is None:
            topo_id = topo_counter
        topo_counter += 1

        for entry in entries:
            frames = entry.get("frames", [])
            pid_actions = entry.get("pid_actions", [])
            if len(frames) != len(pid_actions):
                n = min(len(frames), len(pid_actions))
                frames = frames[:n]
                pid_actions = pid_actions[:n]

            track_id = run_uid_counter
            run_uid_counter += 1

            dropped = 0
            for i, (frame_name, action) in enumerate(zip(frames, pid_actions)):
                try:
                    steer, throttle = _coerce_action(action)
                except Exception as e:
                    print(f"[Data Conversion] drop frame (action parse): {e}")
                    dropped += 1
                    continue

                src_img = image_dir / f"{frame_name}.jpg"
                if not src_img.exists():
                    print(f"[Data Conversion] missing image: {src_img}")
                    dropped += 1
                    continue

                dst_img = out_pairs_dir / f"image_{start_idx:06d}.jpg"
                dst_js  = out_pairs_dir / f"record_{start_idx:06d}.json"

                dst_img.write_bytes(src_img.read_bytes())
                write_frame_record(dst_js, steer, throttle, track_id, topo_id, i)
                start_idx += 1

            print(f"[Data Conversion:ok] {jf.name}: topo_id={topo_id} track_id={track_id} dropped={dropped}")

    print(f"[Data Conversion] wrote data at {out_pairs_dir}")
