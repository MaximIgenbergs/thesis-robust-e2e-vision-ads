# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime, json, os
from pathlib import Path
from typing import Dict, Iterable, Optional
import matplotlib
matplotlib.use("Agg")  # headless
from matplotlib import pyplot as plt

def _repo_root_from(file: str, up: int = 4) -> Path:
    p = Path(file).resolve()
    for _ in range(up):
        p = p.parent
    return p

def resolve_runs_dir(model_key: str, fallback_subdir: str, file: str) -> Path:
    """
    Try to reuse sims.udacity.configs.paths.TRAINING_RUNS_DIR if present.
    Else fall back to <repo_root>/runs/<fallback_subdir>.
    """
    try:
        from sims.udacity.configs.paths import TRAINING_RUNS_DIR as _TR  # type: ignore
        base = Path(_TR)  # respect user's existing location
    except Exception:
        base = _repo_root_from(file) / "runs" / fallback_subdir
    base.mkdir(parents=True, exist_ok=True)
    return base

def make_run_dir(base: Path, model_key: str) -> Path:
    run_id = datetime.datetime.now().strftime(f"{model_key}_run_%Y-%m-%d_%H-%M-%S")
    rd = base / run_id
    rd.mkdir(parents=True, exist_ok=True)
    return rd

def write_meta(run_dir: Path, meta: Dict):
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

def loss_plot(history, run_dir: Path, title: str = "Training & Validation Loss") -> Path:
    plt.figure()
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title); plt.legend(); plt.tight_layout()
    out = run_dir / "loss_curve.png"
    plt.savefig(out); plt.close()
    return out
