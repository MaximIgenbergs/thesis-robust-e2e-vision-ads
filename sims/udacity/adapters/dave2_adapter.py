# sims/udacity/adapters/dave2_adapter.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from sims.udacity.checkpoints.dave2.model import build_model as build_dave2
from perturbationdrive import ADS


class Dave2Adapter(ADS):
    """
    Dave-2 adapter for PerturbationDrive.

    Implements ADS abstract interface:
      - name()   -> str
      - reset()  -> None
      - action(o)-> control tuple (steer, throttle, brake)

    We also expose predict(rgb) and __call__(rgb) as convenience wrappers.
    """

    _INPUT_SHAPE: Tuple[int, int, int] = (66, 200, 3)

    def __init__(
        self,
        weights: Optional[Path] = None,
        image_size_hw: Tuple[int, int] = (66, 200),
        device: Optional[str] = None,
        normalize: str = "imagenet",
    ) -> None:
        super().__init__()
        self._name = "dave2"
        self.model = self._load_model(weights)

    # ---- ADS required API ----------------------------------------------------

    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        # stateless model; nothing to clear
        return

    def action(self, observation: np.ndarray) -> Tuple[float, float, float]:
        """
        PerturbationDrive calls this to get the control action for a single RGB frame.
        We delegate to predict() so all preprocessing stays in one place.
        """
        return self.predict(observation)

    # ---- Convenience API -----------------------------------------------------

    def __call__(self, rgb: np.ndarray) -> Tuple[float, float, float]:
        return self.predict(rgb)

    def predict(self, rgb: np.ndarray) -> Tuple[float, float, float]:
        """
        Args:
            rgb: HxWx3 uint8 RGB frame from the simulator.
        Returns:
            (steer, throttle, brake) floats
        """
        x = self._preprocess(rgb)                  # (1, 66, 200, 3), float32 in [-1, 1]
        y = self.model(x, training=False).numpy()  # (1, 2)
        steer = float(y[0, 0])
        throttle = float(y[0, 1])
        brake = 0.0
        return steer, throttle, brake

    # ---- Internals -----------------------------------------------------------

    def _load_model(self, weights: Optional[Path]) -> tf.keras.Model: # type: ignore
        if weights is not None:
            wpath = str(weights)
            try:
                return tf.keras.models.load_model(wpath, compile=False)
            except Exception:
                m = build_dave2(input_shape=self._INPUT_SHAPE)
                m.load_weights(wpath)
                return m
        else:
            return build_dave2(input_shape=self._INPUT_SHAPE)

    def _preprocess(self, rgb: np.ndarray) -> tf.Tensor:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)
        t = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        t = tf.image.resize(t, size=self._INPUT_SHAPE[:2],
                            method=tf.image.ResizeMethod.BILINEAR)
        t = tf.cast(t, tf.float32) / 127.5 - 1.0
        t = tf.expand_dims(t, axis=0)  # (1, H, W, C)
        return t
