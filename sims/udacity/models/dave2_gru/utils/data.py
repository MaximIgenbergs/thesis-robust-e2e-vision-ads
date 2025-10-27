# -*- coding: utf-8 -*-
"""
Sequence-aware data utilities for DAVE2-GRU.

- Consumes the same per-frame files as PD's example trainer:
  * image:  x_cam-image_array_.jpg
  * label:  record_x.json  with fields:
            { "user/angle": <float>, "user/throttle": <float>, "meta/track_id": <int>, "meta/frame": <int> }

- For GRU we require contiguous frames from the same track/session.
- We build sliding windows of length T with stride S (default S=1).
- No shuffling within a track; optional global shuffle of sequence order is disabled by default.

If 'meta/track_id' is missing, we fallback to:
- parse the parent directory if it contains 'track_<id>' OR
- treat everything as track_id=0.

Author: Maxim Igenbergs. Backbone format inspired by PerturbationDrive examples.
"""

from __future__ import annotations
import json, os, fnmatch
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# add project root to path
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse base DAVE-2 config if available
try:
    from sims.udacity.models.dave2.config import INPUT_SHAPE as _INPUT_SHAPE
    ROW, COL, CH = _INPUT_SHAPE
except Exception:
    ROW, COL, CH = 120, 160, 3


def _load_json(path: str) -> dict:
    with open(path, "rt") as f:
        return json.load(f)


def _infer_track_id(img_path: str, meta: dict) -> int:
    # 1) prefer JSON
    tid = None
    for k in ("meta/track_id", "track_id", "meta:track_id"):
        if k in meta:
            tid = meta[k]
            break
    if tid is not None:
        try:
            return int(tid)
        except Exception:
            pass

    # 2) parent directory like .../track_12/...
    parent = os.path.basename(os.path.dirname(img_path))
    if parent.startswith("track_"):
        try:
            return int(parent.split("_", 1)[1])
        except Exception:
            pass

    # 3) fallback
    return 0


def _frame_index(img_path: str, meta: dict) -> int:
    # Prefer JSON explicit frame index
    for k in ("meta/frame", "frame", "frame_index"):
        if k in meta:
            try:
                return int(meta[k])
            except Exception:
                pass

    # Parse from filename pattern '*_<index>.jpg'
    base = os.path.basename(img_path)
    stem, _ = os.path.splitext(base)
    try:
        idx = int(stem.split("_")[-1])
        return idx
    except Exception:
        return -1  # unknown


def _find_files(glob_mask: str) -> List[str]:
    glob_mask = os.path.expanduser(glob_mask)
    base, mask = os.path.split(glob_mask)
    out: List[str] = []
    for root, _, files in os.walk(base):
        for fn in fnmatch.filter(files, mask):
            if fn.lower().endswith(".jpg"):
                out.append(os.path.join(root, fn))
    return out


def _load_image(path: str) -> np.ndarray:
    im = Image.open(path)
    # Resize if needed to match (ROW, COL)
    if im.size != (COL, ROW):
        im = im.resize((COL, ROW), resample=Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def build_sequences(inputs_glob: str,
                    seq_len: int,
                    stride: int = 1,
                    expect_outputs: int = 2,
                    drop_last_incomplete: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scans images and JSONs, groups by track_id, sorts by frame index, and emits (X, y):

    X.shape = (N, T, ROW, COL, CH)
    y.shape = (N, expect_outputs)   # labels taken from the LAST frame in each sequence

    We DO NOT shuffle frames; sequences are contiguous within each track.
    """
    img_paths = _find_files(inputs_glob)
    if not img_paths:
        return np.empty((0, seq_len, ROW, COL, CH), np.float32), np.empty((0, expect_outputs), np.float32)

    # Collect per-track ordered frame records
    by_track: Dict[int, List[Tuple[int, str, dict]]] = {}
    for img in img_paths:
        # derive paired JSON path
        frame_no = os.path.basename(img).split("_")[-1].split(".")[0]
        js = os.path.join(os.path.dirname(img), f"record_{frame_no}.json")
        if not os.path.exists(js):
            continue
        meta = _load_json(js)

        tid = _infer_track_id(img, meta)
        idx = _frame_index(img, meta)
        by_track.setdefault(tid, []).append((idx, img, meta))

    # Sort each track by frame index
    for tid in by_track:
        by_track[tid].sort(key=lambda tup: tup[0])

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for _, frames in by_track.items():
        # sliding windows: contiguous and same track by construction
        N = len(frames)
        if N < seq_len:
            continue
        end = N - (0 if drop_last_incomplete else (seq_len - 1))
        for start in range(0, end, stride):
            window = frames[start:start + seq_len]
            if len(window) < seq_len:
                break

            imgs = [_load_image(p) for (_, p, __) in window]
            # Super important: label from LAST frame of the window
            _, _, last_meta = window[-1]

            steer = float(last_meta.get("user/angle", last_meta.get("user/angel", 0.0)))
            throttle = float(last_meta.get("user/throttle", 0.0))
            if expect_outputs == 1:
                y = np.array([steer], dtype=np.float32)
            else:
                y = np.array([steer, throttle], dtype=np.float32)

            Xs.append(np.stack(imgs, axis=0))  # (T,H,W,C)
            ys.append(y)

    if not Xs:
        return np.empty((0, seq_len, ROW, COL, CH), np.float32), np.empty((0, expect_outputs), np.float32)

    X = np.stack(Xs, axis=0).astype(np.float32)
    Y = np.stack(ys, axis=0).astype(np.float32)
    return X, Y
