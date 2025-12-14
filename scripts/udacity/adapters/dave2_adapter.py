from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from perturbationdrive import ADS
from scripts.udacity.models.dave2.model import build_dave2


class Dave2Adapter(ADS):

    def __init__(self, weights: Optional[Path] = None, image_size_hw: tuple[int, int] = (66, 200), device: Optional[str] = None, normalize: str = "imagenet") -> None:
        super().__init__()
        self._name = "dave2"
        self.device = device
        self._normalize = normalize

        h, w = image_size_hw
        self._input_shape = (int(h), int(w), 3)

        self.model = self.load_model(weights)

    # ADS interface

    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        return  # not a stateful model, function will not be used

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
        Run the model on an image and return [[steer, throttle]].
        """
        x = self.preprocess(image)
        y = self.model(x, training=False).numpy()

        steer = float(y[0, 0])
        throttle = float(y[-1, 1])
        return np.asarray([[steer, throttle]], dtype=np.float32)

    def load_model(self, weights: Optional[Path]) -> tf.keras.Model:  # type: ignore
        """
        Build the model for the configured input shape and load weights if a path is given.
        """
        wpath = str(weights)
        try:
            return tf.keras.models.load_model(wpath, compile=False)
        except Exception:
            print(f"[scripts:adapter:dave2][WARN] Failed to load model from {wpath}. Ensure the model architecture matches the weights.")
            raise

    def preprocess(self, image: np.ndarray) -> tf.Tensor:
        """
        Resize an image to match the training format.
        The model's own Lambda layer (t/255.0) does the normalization.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)

        target_h, target_w, _ = self._input_shape

        t = tf.convert_to_tensor(image, dtype=tf.float32)
        t = tf.image.resize(
            t,
            size=(target_h, target_w),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        t = tf.expand_dims(t, axis=0)
        return t
