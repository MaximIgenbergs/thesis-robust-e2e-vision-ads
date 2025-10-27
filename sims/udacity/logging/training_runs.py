# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime, json
from pathlib import Path
from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg")  # headless backend
from matplotlib import pyplot as plt


def make_run_dir(model_key: str, map_name: str) -> Path:
    """
    Create <CKPTS_DIR>/<model_key>/<map_name>_<YYYYMMDD-HHMMSS>/
    and return the created Path.
    """
    from sims.udacity.configs.paths import CKPTS_DIR  # imported here to avoid circular import

    base = Path(CKPTS_DIR).expanduser().resolve() / model_key
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{map_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_meta(run_dir: Path, meta: Dict) -> None:
    """Write meta.json inside run_dir."""
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def loss_plot(history, run_dir: Path, title: str = "Training & Validation Loss") -> Path:
    """Plot and save training/validation loss curves."""
    plt.figure()
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out = run_dir / "loss_curve.png"
    plt.savefig(out)
    plt.close()
    return out
