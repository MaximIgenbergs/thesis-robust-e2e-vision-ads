# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np
import tensorflow as tf

from sims.udacity.checkpoints.dave2_gru.model import build_dave2_gru
from sims.udacity.checkpoints.dave2_gru.config import INPUT_SHAPE, SEQ_LEN  # (H, W, C), T
from perturbationdrive import ADS


class Dave2GRUAdapter(ADS):
    """
    DAVE-2-GRU adapter for PerturbationDrive (sequence-based).

    Maintains a rolling window of the last SEQ_LEN frames. On cold start,
    fills the buffer by repeating the first frame so the very first call
    already has a full sequence.
    """

    def __init__(
        self,
        weights: Optional[Path] = None,
        image_size_hw: Tuple[int, int] = (INPUT_SHAPE[0], INPUT_SHAPE[1]),
        device: Optional[str] = None,
        normalize: str = "imagenet",  # accepted for compatibility, not used
    ) -> None:
        super().__init__()
        self._name = "dave2_gru"
        self._in_hwc = INPUT_SHAPE
        self._T = SEQ_LEN
        self._device = device
        self._buf: Deque[np.ndarray] = deque(maxlen=self._T)
        self.model = self._load_model(weights)

    # ---- PD interface ----

    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._buf.clear()

    def action(self, observation: np.ndarray) -> np.ndarray:
        return self.predict(observation)


    # ---- Convenience ----

    def __call__(self, rgb: np.ndarray) -> Tuple[float, float, float]:
        return self.predict(rgb)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        frame = self._preprocess_frame(rgb)
        if not self._buf:
            for _ in range(self._T):
                self._buf.append(frame.copy())
        else:
            self._buf.append(frame)

        seq = np.stack(list(self._buf), axis=0)              # (T,H,W,C)
        x = tf.convert_to_tensor(seq[np.newaxis, ...])       # (1,T,H,W,C)
        y = self.model(x, training=False).numpy()

        if y.shape[-1] == 1:
            steer = float(y[0, 0]); throttle = 0.30
        else:
            steer, throttle = float(y[0, 0]), float(y[0, 1])

        return np.asarray([[steer, throttle]], dtype=np.float32)


    # ---- Internals ----

    def _load_model(self, weights: Optional[Path]) -> tf.keras.Model: # type: ignore
        # Try loading a SavedModel/H5 first; otherwise, build and load weights.
        if weights is not None:
            wpath = str(weights)
            try:
                return tf.keras.models.load_model(wpath, compile=False)
            except Exception:
                m = build_dave2_gru(seq_len=self._T)  # num_outputs from training default
                m.load_weights(wpath)
                return m
        return build_dave2_gru(seq_len=self._T)

    def _preprocess_frame(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)
        t = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        t = tf.image.resize(t, size=self._in_hwc[:2], method=tf.image.ResizeMethod.BILINEAR)
        t = tf.cast(t, tf.float32)  # keep 0..255; model's internal Lambda does /255
        return t.numpy()
