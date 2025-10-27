from __future__ import annotations
import fnmatch, json, os, random
from typing import List, Tuple, Iterator
import numpy as np
from PIL import Image
from ..config import INPUT_SHAPE, NUM_OUTPUTS, VAL_SPLIT, RANDOM_SEED, BATCH_SIZE

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

def _load_image(path: str) -> np.ndarray:
    row, col, ch = INPUT_SHAPE
    im = Image.open(path)
    if im.size != (col, row):
        im = im.resize((col, row), resample=Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

def _load_label(js_path: str) -> List[float]:
    with open(js_path, "r") as f:
        d = json.load(f)
    # tolerate PD's 'user/angel' typo
    steer = float(d.get("user/angle", d.get("user/angel", 0.0)))
    throttle = float(d.get("user/throttle", 0.0))
    return [steer] if NUM_OUTPUTS == 1 else [steer, throttle]

def _shuffle_in_place(lst: List[str]) -> None:
    random.Random(RANDOM_SEED).shuffle(lst)

def make_filelists(inputs_glob: str, val_split: float = VAL_SPLIT) -> Tuple[List[str], List[str]]:
    imgs = _find_images(inputs_glob)
    if not imgs:
        return [], []
    _shuffle_in_place(imgs)  # PD example shuffles before split
    n_val = max(1, int(len(imgs) * val_split))
    return imgs[n_val:], imgs[:n_val]

def data_generator(files: List[str], batch_size: int = BATCH_SIZE) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    row, col, ch = INPUT_SHAPE
    i = 0
    while True:
        batch_imgs, batch_labels = [], []
        for _ in range(batch_size):
            if i >= len(files):
                i = 0
                _shuffle_in_place(files)
            img_path = files[i]; i += 1
            js_path = _json_for(img_path)
            if not os.path.exists(js_path):
                continue
            try:
                x = _load_image(img_path)
                y = _load_label(js_path)
            except Exception:
                continue
            batch_imgs.append(x)
            batch_labels.append(y)
        if batch_imgs:
            X = np.asarray(batch_imgs, dtype=np.float32)
            y = np.asarray(batch_labels, dtype=np.float32)
            yield X, y
