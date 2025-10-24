# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

from sims.udacity.checkpoints.dave2.model import build_model as build_dave2
from sims.udacity.checkpoints.dave2.config import INPUT_SHAPE as FALLBACK_INPUT  # only as fallback
from perturbationdrive import ADS


class Dave2Adapter(ADS):
    def __init__(
        self,
        weights: Optional[Path] = None,
        image_size_hw: Tuple[int, int] = (FALLBACK_INPUT[0], FALLBACK_INPUT[1]),
        device: Optional[str] = None,
        normalize: str = "imagenet",
    ) -> None:
        super().__init__()
        self._name = "dave2"
        self._device = device
        self.model = self._load_model(weights)
        self._in_hwc = self._infer_input_shape(self.model) or FALLBACK_INPUT
        print(f"[Dave2Adapter] Using model input shape: {self._in_hwc}")

    # ---- PD interface ----
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        return

    def action(self, observation: np.ndarray) -> np.ndarray:
        return self.predict(observation)

    # ---- Inference ----
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        x = self._preprocess(rgb)                      # (1, H, W, C) matching model
        y = self.model(x, training=False).numpy()      # (1, O)

        if y.shape[-1] == 1:
            steer = float(y[0, 0]); throttle = 0.30
        else:
            steer, throttle = float(y[0, 0]), float(y[0, 1])

        # Udacity sim expects batched shape (1, 2)
        return np.asarray([[steer, throttle]], dtype=np.float32)

    # ---- Internals ----
    def _load_model(self, weights: Optional[Path]) -> tf.keras.Model: # type: ignore
        if weights is not None:
            wpath = str(weights)
            try:
                # SavedModel/H5 — preserves original input shape
                return tf.keras.models.load_model(wpath, compile=False)
            except Exception:
                pass
        # Fallback: build in-repo model, then load weights if provided
        try:
            m = build_dave2(input_shape=FALLBACK_INPUT)
        except TypeError:
            m = build_dave2()
        if weights is not None:
            m.load_weights(wpath)
        return m

    def _infer_input_shape(self, model: tf.keras.Model): # type: ignore
        try:
            shp = model.input_shape  # e.g., (None, H, W, C) or list thereof
            if isinstance(shp, (list, tuple)) and isinstance(shp[0], (list, tuple)):
                shp = shp[0]
            return (int(shp[1]), int(shp[2]), int(shp[3]))
        except Exception:
            return None

    def _preprocess(self, rgb: np.ndarray) -> tf.Tensor:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)
        t = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        t = tf.image.resize(t, size=self._in_hwc[:2], method=tf.image.ResizeMethod.BILINEAR)
        t = tf.cast(t, tf.float32)  # model's internal Lambda does /255.0
        t = tf.expand_dims(t, axis=0)
        return t
