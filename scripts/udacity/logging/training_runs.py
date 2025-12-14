from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from matplotlib import pyplot as plt

from scripts import CKPTS_DIR


def make_run_dir(model_key: str, map_name: str) -> Path:
    """
    Create <CKPTS_DIR>/<model_key>/<map_name>_<YYYYMMDD-HHMMSS>/.
    """
    base = Path(CKPTS_DIR).expanduser().resolve() / model_key
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{map_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    text = json.dumps(meta, indent=2, sort_keys=True)
    (run_dir / "meta.json").write_text(text, encoding="utf-8")


def loss_plot(history, run_dir: Path, title: str = "Training & Validation Loss") -> Path:
    plt.figure()

    if "loss" in history.history:
        plt.plot(history.history["loss"], label="loss")

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
