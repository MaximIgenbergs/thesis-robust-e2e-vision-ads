# -*- coding: utf-8 -*-
"""
io_utils.py — background image I/O and metadata dump.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


class AsyncImageWriter:
    """
    Background encoder for CARLA images with counters.

    Parameters
    ----------
    max_queue : int
        Max number of pending frames before dropping or blocking.
    drop_if_full : bool
        If True, drops new frames when queue is full; else blocks.
    image_format : {'png','jpg'}
        Output format.
    jpeg_quality : int
        JPEG quality (only for 'jpg').
    png_compress_level : int
        PNG compress_level 0..9 (only for 'png').
    """
    def __init__(
        self,
        max_queue: int = 128,
        drop_if_full: bool = True,
        image_format: str = "png",
        jpeg_quality: int = 90,
        png_compress_level: int = 6,
    ):
        self.q: "queue.Queue[Tuple[str, bytes, int, int]]" = queue.Queue(max_queue)
        self.drop_if_full = drop_if_full
        self.image_format = image_format.lower()
        self.jpeg_quality = int(jpeg_quality)
        self.png_compress_level = int(png_compress_level)

        self.submitted = 0
        self.written = 0
        self.dropped = 0

        self._alive = True
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def submit(self, path: str, carla_image) -> None:
        payload = (path, bytes(carla_image.raw_data), carla_image.width, carla_image.height)
        try:
            self.q.put_nowait(payload)
            self.submitted += 1
        except queue.Full:
            if self.drop_if_full:
                self.dropped += 1
            else:
                self.q.put(payload)
                self.submitted += 1

    def _worker(self) -> None:
        while self._alive or not self.q.empty():
            try:
                path, buf, w, h = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3][:, :, ::-1]  # BGRA->RGB
                img = Image.fromarray(arr)

                if self.image_format == "jpg" or self.image_format == "jpeg":
                    img.save(path, format="JPEG", quality=self.jpeg_quality, optimize=True)
                else:
                    img.save(path, format="PNG", compress_level=self.png_compress_level, optimize=True)

                self.written += 1
            except Exception:
                # swallow and continue; increase dropped? No: submitted already counted.
                pass
            finally:
                self.q.task_done()

    def close(self, return_stats: bool = False, wait: bool = True):
        self._alive = False
        if wait:
            try:
                self.q.join()
                self._t.join(timeout=4)
            except Exception:
                pass
        stats = {"submitted": self.submitted, "written": self.written, "dropped": self.dropped}
        return stats if return_stats else None


def _git_sha_or_none(cwd: Optional[str] = None) -> Optional[str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        return sha.decode("utf-8").strip()
    except Exception:
        return None


def write_run_metadata(
    out_dir: str,
    world,
    tm_port: int,
    cfg: Dict,
    run_id: str,
    host: str,
    rpc_port: int,
) -> None:
    """Dump a JSON with reproducibility-critical metadata."""
    try:
        town_full = world.get_map().name
    except Exception:
        town_full = cfg.get("town", "UnknownTown")

    try:
        ws = world.get_settings()
        world_settings = {
            "synchronous_mode": bool(ws.synchronous_mode),
            "fixed_delta_seconds": float(ws.fixed_delta_seconds) if ws.fixed_delta_seconds else None,
            "max_substeps": getattr(ws, "max_substeps", None),
            "max_substep_delta_time": getattr(ws, "max_substep_delta_time", None),
        }
    except Exception:
        world_settings = {}

    try:
        carla_version = getattr(__import__("carla"), "__version__", "unknown")
    except Exception:
        carla_version = "unknown"

    # camera intrinsics/extrinsics (as given; sensor transform is fixed by config)
    camera_meta = dict(cfg.get("camera", {}))

    meta = {
        "schema_version": "1.0",
        "run_id": run_id,
        "started_utc": datetime_utc_iso(),
        "host": host,
        "rpc_port": rpc_port,
        "tm_port": tm_port,
        "carla_version": carla_version,
        "map_name": town_full,
        "seed": int(cfg["seed"]),
        "fps": int(cfg["fps"]),
        "runtime": cfg.get("runtime", {}),
        "logging": cfg.get("logging", {}),
        "weather": cfg.get("weather", {}),
        "routes": cfg.get("routes", {}),
        "ego_blueprint": cfg.get("ego_blueprint"),
        "traffic_vehicles": cfg.get("traffic_vehicles"),
        "world_settings": world_settings,
        "camera": camera_meta,
        "config": cfg,  # embed whole config for completeness
        "git_sha": _git_sha_or_none(cwd=os.path.dirname(out_dir)),
    }

    path = os.path.join(out_dir, "run_meta.json")
    try:
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    except Exception:
        pass


def datetime_utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
