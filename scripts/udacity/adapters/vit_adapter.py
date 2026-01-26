from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from perturbationdrive import ADS
from scripts.udacity.models.vit.model import ViT
from scripts.udacity.models.vit.config import INPUT_SHAPE, NUM_OUTPUTS


class ViTAdapter(ADS):

    def __init__(self, weights: Optional[Path] = None, image_size_hw: Tuple[int, int] = (160, 160), device: Optional[str] = None) -> None:
        super().__init__()
        self._name = "vit"

        h, w = image_size_hw
        self._input_shape = (int(h), int(w), 3)

        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        self._to_tensor = T.ToTensor()
        self.model = self.load_model(weights)

    # ADS interface

    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        return  # stateless model

    def action(self, observation: np.ndarray) -> np.ndarray:
        return self.predict(observation)

    # Convenience

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.predict(frame)

    # Core

    def load_model(self, weights: Optional[Path]) -> torch.nn.Module:
        if weights is None:
            raise ValueError("[scripts:adapter:vit][ERR] weights path must not be None for ViTAdapter.")

        h, w, _ = self._input_shape
        model = ViT(input_shape=(3, h, w), learning_rate=0.0)

        wpath = str(weights)
        try:
            state = torch.load(wpath, map_location=self.device)
            model.load_state_dict(state)
        except Exception:
            print(f"[scripts:adapter:vit][WARN] Failed to load state_dict from {wpath}. Ensure the ViT architecture matches the checkpoint.")
            raise

        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Resize and convert an image to a (1, 3, H, W) float32 tensor in [0, 1].
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)

        target_h, target_w, _ = self._input_shape

        pil = Image.fromarray(image).convert("RGB")
        if pil.size != (target_w, target_h):
            pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)

        x = self._to_tensor(pil)  # (3, H, W), float32 in [0, 1]
        x = x.unsqueeze(0)        # (1, 3, H, W)
        return x.to(self.device, non_blocking=True)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run the ViT model on an image and return [[steer, throttle]].
        """
        x = self.preprocess(image)

        with torch.no_grad():
            y = self.model(x)

        y_np = y.detach().cpu().numpy().reshape(-1)

        steer = float(y_np[0]) if y_np.size > 0 else 0.0
        if NUM_OUTPUTS > 1 and y_np.size > 1:
            throttle = float(y_np[1])
        else:
            throttle = 0.0

        return np.asarray([[steer, throttle]], dtype=np.float32)
