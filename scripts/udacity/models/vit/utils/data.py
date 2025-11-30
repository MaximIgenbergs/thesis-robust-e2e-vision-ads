from __future__ import annotations

import json
import os
import random
from typing import Iterator, List, Tuple

import torch
from PIL import Image
import torchvision.transforms as T

from scripts.udacity.models.vit.config import INPUT_SHAPE, NUM_OUTPUTS, VAL_SPLIT, RANDOM_SEED, BATCH_SIZE


def find_images(data_dir: str) -> List[str]:
    data_dir = os.path.expanduser(data_dir)
    hits: List[str] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(".jpg"):
                hits.append(os.path.join(root, name))
    hits.sort()
    return hits


def json_path_for(image_path: str) -> str:
    base = os.path.basename(image_path)
    idx = os.path.splitext(base)[0].split("_")[-1]
    return os.path.join(os.path.dirname(image_path), f"record_{idx}.json")


def load_image(path: str) -> Image.Image:
    row, col, ch = INPUT_SHAPE
    im = Image.open(path).convert("RGB")
    if im.size != (col, row):
        im = im.resize((col, row), resample=Image.BILINEAR)
    return im


def load_label(json_path: str) -> list[float]:
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    steer = float(d.get("user/angle", d.get("user/angel", 0.0)))
    throttle = float(d.get("user/throttle", 0.0))

    if NUM_OUTPUTS == 1:
        return [steer]

    return [steer, throttle]


def shuffle_in_place(files: List[str]) -> None:
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(files)


def make_filelists(data_dir: str, val_split: float = VAL_SPLIT) -> Tuple[List[str], List[str]]:
    images = find_images(data_dir)
    if not images:
        return [], []

    shuffle_in_place(images)

    n_val = max(1, int(len(images) * val_split))
    val_files = images[:n_val]
    train_files = images[n_val:]
    return train_files, val_files


_AUGMIX = getattr(T, "AugMix", None)

if _AUGMIX is not None:
    _train_transform = T.Compose([
        _AUGMIX(),
        T.ToTensor(),
    ])
else:
    _train_transform = T.ToTensor()

_val_transform = T.ToTensor()


def random_flip(img: Image.Image, y: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
    if random.random() > 0.5:
        img = T.functional.hflip(img)
        y = y.clone()
        if y.numel() > 0:
            y[0] = -y[0]
    return img, y


def data_generator(files: List[str], batch_size: int = BATCH_SIZE, train: bool = True) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    i = 0
    n = len(files)
    if n == 0:
        raise RuntimeError("data_generator called with empty file list.")

    max_empty_batches = 2 * n

    while True:
        batch_images: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        empty_loops = 0

        while len(batch_images) < batch_size:
            if i >= n:
                i = 0
                shuffle_in_place(files)

            image_path = files[i]
            i += 1

            json_path = json_path_for(image_path)
            if not os.path.exists(json_path):
                empty_loops += 1
                if empty_loops > max_empty_batches:
                    raise RuntimeError("No valid (image, json) pairs found in data_generator.")
                continue

            try:
                img = load_image(image_path)
                y_vals = load_label(json_path)
                y = torch.tensor(y_vals, dtype=torch.float32)
                if train:
                    img, y = random_flip(img, y)
                x = (_train_transform if train else _val_transform)(img)
            except Exception as e:
                empty_loops += 1
                if empty_loops > max_empty_batches:
                    raise RuntimeError(f"Repeated failures in data_generator. Last error: {e}")
                continue

            batch_images.append(x)
            batch_labels.append(y)

        x_batch = torch.stack(batch_images, dim=0)
        y_batch = torch.stack(batch_labels, dim=0)
        yield x_batch, y_batch
