from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import tensorflow as tf

from perturbationdrive import ADS
from scripts.udacity.models.dave2_gru.model import build_dave2_gru


class Dave2GRUAdapter(ADS):

    def __init__(self, weights: Optional[Path] = None, image_size_hw: tuple[int, int] = (66, 200), seq_len: int = 8, device: Optional[str] = None, normalize: str = "imagenet") -> None:
        super().__init__()
        self._name = "dave2_gru"
        self.device = device
        self._normalize = normalize

        h, w = image_size_hw
        self._input_shape = (int(h), int(w), 3)
        self.seq_len = int(seq_len)

        self.buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.model = self.load_model(weights)

    # ADS interface

    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self.buffer.clear()

    def action(self, observation: np.ndarray) -> np.ndarray:
        """
        Return the control command for one image as [[steer, throttle]].
        """
        return self.predict(observation)

    # Functions

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.predict(frame)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run the sequence model on an image and return [[steer, throttle]].
        """
        frame = self.preprocess_frame(image)

        if not self.buffer:
            for _ in range(self.seq_len):
                self.buffer.append(frame.copy())
        else:
            self.buffer.append(frame)

        seq = np.stack(list(self.buffer), axis=0)  # (T, H, W, C)
        x = tf.convert_to_tensor(seq[np.newaxis, ...])  # (1, T, H, W, C)
        y = self.model(x, training=False).numpy()

        if y.shape[-1] == 1:
            steer = float(y[0, 0])
            throttle = 0.30
        else:
            steer = float(y[0, 0])
            throttle = float(y[0, 1])

        return np.asarray([[steer, throttle]], dtype=np.float32)

    def load_model(self, weights: Optional[Path]) -> tf.keras.Model:  # type: ignore
        """
        Build the model for the configured input shape and sequence length, then load weights.
        """
        if weights is None:
            return build_dave2_gru(input_shape=self._input_shape, seq_len=self.seq_len)

        wpath = str(weights)
        try:
            return tf.keras.models.load_model(wpath, compile=False)
        except Exception:
            model = build_dave2_gru(input_shape=self._input_shape, seq_len=self.seq_len)
            model.load_weights(wpath)
            return model

    def preprocess_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Resize and cast an image to match the model's expected input format.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)

        target_h, target_w, _ = self._input_shape

        t = tf.convert_to_tensor(image, dtype=tf.uint8)
        t = tf.image.resize(
            t,
            size=(target_h, target_w),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        t = tf.cast(t, tf.float32)
        return t.numpy()
