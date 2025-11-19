from __future__ import annotations
import fnmatch
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _find_images(glob_mask: str) -> List[str]:
    glob_mask = os.path.expanduser(glob_mask)
    base, mask = os.path.split(glob_mask)
    hits: List[str] = []
    for root, _, files in os.walk(base):
        for fn in fnmatch.filter(files, mask):
            if fn.lower().endswith(".jpg"):
                hits.append(os.path.join(root, fn))
    hits.sort()
    return hits


def _json_for(img_path: str) -> str:
    base = os.path.basename(img_path)
    idx = os.path.splitext(base)[0].split("_")[-1]
    return os.path.join(os.path.dirname(img_path), f"record_{idx}.json")


def make_filelists(
    inputs_glob: str,
    val_split: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Identical spirit to DAVE2: shuffle then split into train / val lists of image paths.
    """
    imgs = _find_images(inputs_glob)
    if not imgs:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_split))
    return imgs[n_val:], imgs[:n_val]


class UdacityImageDataset(Dataset):
    """
    PyTorch Dataset for Udacity-style frames:

      image_000001.jpg
      record_000001.json  with:
        "user/angle": <float>,
        "user/throttle": <float>,
        ...

    Returns:
      img:    FloatTensor (3, H, W) (after transform)
      target: FloatTensor (2,) [steering, throttle]
    """

    def __init__(self, files: List[str], img_size: Tuple[int, int] = (224, 224)):
        self.files = files
        self.img_size = img_size

        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img_path = self.files[idx]
        js_path = _json_for(img_path)
        if not os.path.exists(js_path):
            raise FileNotFoundError(f"Missing JSON for {img_path}: {js_path}")

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)

        # Load label
        with open(js_path, "r") as f:
            meta = json.load(f)

        # tolerate PD's 'user/angel' typo
        steer = float(meta.get("user/angle", meta.get("user/angel", 0.0)))
        throttle = float(meta.get("user/throttle", 0.0))

        y = torch.tensor([steer, throttle], dtype=torch.float32)
        return img_t, y
