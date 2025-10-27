# -*- coding: utf-8 -*-
from __future__ import annotations
import fnmatch, json, os
import random
from typing import Dict, Iterable, Iterator, List, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf

# Small, fast JSON & image helpers -------------------------------------------

def _load_json(path: str) -> dict:
    with open(path, "rt") as f:
        return json.load(f)

def _json_for(img_path: str) -> str:
    base = os.path.basename(img_path)
    idx = os.path.splitext(base)[0].split("_")[-1]
    return os.path.join(os.path.dirname(img_path), f"record_{idx}.json")

def _frame_index(img_path: str, meta: dict) -> int:
    for k in ("meta/frame", "frame", "frame_index"):
        if k in meta:
            try:
                return int(meta[k])
            except Exception:
                pass
    # fallback from filename ..._<idx>.jpg
    stem = os.path.splitext(os.path.basename(img_path))[0]
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return -1

def _track_id(img_path: str, meta: dict) -> int:
    for k in ("meta/track_id", "track_id"):
        if k in meta:
            try:
                return int(meta[k])
            except Exception:
                pass
    # parent dir like track_12
    parent = os.path.basename(os.path.dirname(img_path))
    if parent.startswith("track_"):
        try:
            return int(parent.split("_", 1)[1])
        except Exception:
            pass
    return 0

def _find_images(glob_mask: str) -> List[str]:
    glob_mask = os.path.expanduser(glob_mask)
    base, mask = os.path.split(glob_mask)
    out: List[str] = []
    for root, _, files in os.walk(base):
        for fn in fnmatch.filter(files, mask):
            if fn.lower().endswith(".jpg"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out

# Index by track (paths only; no pixels in RAM) -------------------------------

def index_by_track(inputs_glob: str) -> Dict[int, List[Tuple[int, str, str]]]:
    """
    Returns: {track_id: [(frame_idx, img_path, json_path), ...] (sorted by frame_idx)}
    """
    imgs = _find_images(inputs_glob)
    per_track: Dict[int, List[Tuple[int, str, str]]] = {}
    for img in imgs:
        js = _json_for(img)
        if not os.path.exists(js):
            continue
        meta = _load_json(js)
        tid  = _track_id(img, meta)
        idx  = _frame_index(img, meta)
        per_track.setdefault(tid, []).append((idx, img, js))
    # sort each track
    for tid in per_track:
        per_track[tid].sort(key=lambda t: t[0])
    return per_track

def split_tracks(per_track, val_split: float, seed: int = 42):
    tids = sorted(per_track.keys())
    random.Random(seed).shuffle(tids)         # <— shuffle deterministically
    n_val = max(1, int(len(tids) * val_split))
    return tids[:-n_val] if n_val < len(tids) else [], tids[-n_val:]

def count_sequences(n_frames: int, seq_len: int, stride: int) -> int:
    if n_frames < seq_len:
        return 0
    return (n_frames - seq_len) // stride + 1

def total_sequences(per_track: Dict[int, List[Tuple[int,str,str]]],
                    tids: List[int], seq_len: int, stride: int) -> int:
    return sum(count_sequences(len(per_track[tid]), seq_len, stride) for tid in tids)

# Image/label loading per window ----------------------------------------------

def _load_image(path: str, out_hw: Tuple[int,int]) -> np.ndarray:
    H, W = out_hw
    im = Image.open(path)
    if im.size != (W, H):
        im = im.resize((W, H), resample=Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

def _read_label(js_path: str, expect_outputs: int) -> np.ndarray:
    d = _load_json(js_path)
    steer = float(d.get("user/angle", d.get("user/angel", 0.0)))
    throttle = float(d.get("user/throttle", 0.0))
    if expect_outputs == 1:
        return np.asarray([steer], dtype=np.float32)
    return np.asarray([steer, throttle], dtype=np.float32)

# Python generator that yields UNBATCHED windows -------------------------------

def iter_windows(per_track: Dict[int, List[Tuple[int,str,str]]],
                 use_tids: List[int],
                 seq_len: int,
                 stride: int,
                 out_hw: Tuple[int,int],
                 num_outputs: int
                 ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for tid in use_tids:
        frames = per_track[tid]
        N = len(frames)
        if N < seq_len:
            continue
        # sliding windows inside the SAME track
        for start in range(0, N - seq_len + 1, stride):
            window = frames[start:start + seq_len]
            imgs = [_load_image(img, out_hw) for (_, img, __) in window]
            # label from LAST frame in window
            _, _, last_js = window[-1]
            y = _read_label(last_js, num_outputs)
            X = np.stack(imgs, axis=0)  # (T,H,W,C)
            yield X, y

# tf.data wrappers -------------------------------------------------------------

def make_sequence_dataset(per_track, tids, seq_len, stride, out_shape, num_outputs,
                          batch_size, repeat=False) -> tf.data.Dataset:
    H, W, C = out_shape
    gen = lambda: iter_windows(per_track, tids, seq_len, stride, (H, W), num_outputs)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, H, W, C), dtype=tf.float32),
            tf.TensorSpec(shape=(num_outputs,), dtype=tf.float32),
        ),
    )
    if repeat:
        ds = ds.repeat()                  # <— important
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
