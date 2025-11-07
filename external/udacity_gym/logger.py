# PD-compatible episode writers for the legacy udacity_gym stack.

from __future__ import annotations
import json, os, csv, gc
from dataclasses import dataclass, asdict
from typing import List, Optional, Any
import numpy as np
import logging
import sys

try:
    import tensorflow as tf  # for NumpyEncoder parity with PD
except Exception:
    tf = None
from PIL import Image

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if tf is not None and isinstance(obj, tf.Tensor):
            return obj.numpy().tolist() if obj.shape else obj.numpy().item()
        return super().default(obj)

class CSVLogHandler:
    """Minimal drop-in for PD's CSVLogHandler; call .row([...]) to append."""
    def __init__(self, filename: str = "logs.csv", mode: str = "w"):
        self._fh = open(filename, mode, newline="")
        self._wr = csv.writer(self._fh, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    def row(self, values: List[Any]):
        self._wr.writerow(values)
        self._fh.flush()
    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

@dataclass
class ScenarioLite:
    # PD calls these fields .perturbation_function and .perturbation_scale
    perturbation_function: str
    perturbation_scale: int
    # optional road/meta for convenience
    road: Optional[str] = None

@dataclass
class ScenarioOutcomeLite:
    frames: List[int]
    pos: List[List[float]]                  # [[x,y,z], ...]
    xte: List[float]
    speeds: List[float]
    actions: List[List[float]]              # [[steer, throttle], ...]
    pid_actions: List[List[float]]          # keep empty if unused
    scenario: ScenarioLite
    isSuccess: bool
    timeout: bool
    # large arrays kept out of JSON unless images=True
    original_images: Optional[List[np.ndarray]] = None
    perturbed_images: Optional[List[np.ndarray]] = None
    offtrack_count: int = 0
    collision_count: int = 0

class ScenarioOutcomeWriter:
    """Mirror of PD's ScenarioOutcomeWriter with images save option."""
    def __init__(self, file_path: str, overwrite_logs: bool = True):
        self.file_path = file_path
        self._write = True
        if os.path.exists(file_path) and not overwrite_logs:
            print("[Writer] Log path exists; overwrite_logs=False — skipping writes.")
            self._write = False

    def write(self, outcomes: List[ScenarioOutcomeLite], images: bool = False):
        if not self._write or len(outcomes) == 0:
            return
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
        else:
            data = []
        for out in outcomes:
            d = asdict(out)
            pert = d["scenario"]["perturbation_function"]
            sev  = d["scenario"]["perturbation_scale"]
            base = self.file_path.split("logs_")[0] + f"_{pert}_{sev}"
            # remove images from JSON unless saving to disk
            perturbed = d.pop("perturbed_images", None)
            original  = d.pop("original_images",  None)
            if images:
                frames = d.get("frames", list(range(len(perturbed or original or []))))
                if perturbed and len(perturbed) > 0:
                    out_dir = base + "_perturbed"
                    os.makedirs(out_dir, exist_ok=True)
                    for i, img in enumerate(perturbed):
                        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(os.path.join(out_dir, f"{frames[i]}.jpg"))
                else:
                    out_dir = base + "_original"
                    os.makedirs(out_dir, exist_ok=True)
                    for i, img in enumerate(original or []):
                        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(os.path.join(out_dir, f"{frames[i]}.jpg"))
            data.append(d)
            gc.collect()
        with open(self.file_path, "w") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=4)


class CustomLogger:
    def __init__(self, logger_prefix: str):
        self.logger = logging.getLogger(logger_prefix)
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=logging.INFO)

            # FIXME: it seems that it is not needed to stream log to sdtout
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)
            self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)