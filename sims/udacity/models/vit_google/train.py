from __future__ import annotations
import sys
from pathlib import Path
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Add project root
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sims.udacity.models.vit.utils.data import make_filelists, UdacityImageDataset
from sims.udacity.models.vit.model import ViTDriving
from sims.udacity.models.vit.config import MAP_NAME, MODEL_NAME, INPUTS_GLOB, IMG_SIZE, LEARNING_RATE, ALPHA_STEER, VAL_SPLIT, RANDOM_SEED, BATCH_SIZE, EPOCHS, PATIENCE, NUM_WORKERS, BACKBONE_NAME
from sims.udacity.logging.training_runs import make_run_dir, write_meta


def main():
    pl.seed_everything(RANDOM_SEED)
    torch.set_float32_matmul_precision("high")

    train_files, val_files = make_filelists(
        INPUTS_GLOB,
        val_split=VAL_SPLIT,
        seed=RANDOM_SEED,
    )
    if not train_files:
        raise SystemExit(f"[{MODEL_NAME}] No training images found for glob: {INPUTS_GLOB}")

    run_dir = make_run_dir(model_key=MODEL_NAME, map_name=MAP_NAME)
    best_path = run_dir / "best_model.ckpt"

    # Datasets / loaders
    train_ds = UdacityImageDataset(train_files, img_size=IMG_SIZE)
    val_ds   = UdacityImageDataset(val_files,   img_size=IMG_SIZE) if val_files else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    ) if val_ds is not None else None

    # Model
    model = ViTDriving(
        backbone_name=BACKBONE_NAME,
        lr=LEARNING_RATE,
        alpha_steer=ALPHA_STEER,
    )

    # Logging + callbacks
    csv_logger = CSVLogger(
        save_dir=str(run_dir),
        name="pl_logs",
        version="vit_b16",
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best_model",
        monitor="val/loss" if val_loader is not None else "train/loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    earlystop_cb = EarlyStopping(
        monitor="val/loss" if val_loader is not None else "train/loss",
        mode="min",
        patience=PATIENCE,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=EPOCHS,
        logger=csv_logger,
        callbacks=[checkpoint_cb, earlystop_cb],
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save final checkpoint path explicitly
    print(f"[train:{MODEL_NAME}] Best checkpoint: {checkpoint_cb.best_model_path or best_path}")

    # ---- Meta info (compatible with your logging style) ----
    meta = {
        "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "framework": "pytorch-lightning",
        "model": MODEL_NAME,
        "map": MAP_NAME,
        "checkpoint_path": checkpoint_cb.best_model_path or str(best_path),
        "img_size": list(IMG_SIZE),
        "seed": RANDOM_SEED,
        "val_split": VAL_SPLIT,
        "optimizer": {"type": "adamw", "lr": float(LEARNING_RATE)},
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "alpha_steer": float(ALPHA_STEER),
        "backbone": BACKBONE_NAME,
        "data": {"inputs_glob": INPUTS_GLOB},
        "counts": {
            "train_images": int(len(train_files)),
            "val_images": int(len(val_files)),
        },
    }
    write_meta(run_dir, meta)

    # ---- Plot loss curve from metrics.csv (Lightning logs) ----
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path).groupby("epoch").last().reset_index()

        plt.figure()
        if "train/loss" in df.columns:
            plt.plot(df["epoch"], df["train/loss"], label="train_loss")
        if "val/loss" in df.columns:
            plt.plot(df["epoch"], df["val/loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{MODEL_NAME} on {MAP_NAME}")
        plt.legend()

        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        plot_path = run_dir / f"loss_curve_{ts}.png"
        plt.savefig(str(plot_path))
        plt.close()
        print(f"[train:{MODEL_NAME}] Loss curve: {plot_path}")
    else:
        print(f"[train:{MODEL_NAME}] metrics.csv not found at {metrics_path}")

    print(f"[train:{MODEL_NAME}] Run dir: {run_dir}")


if __name__ == "__main__":
    main()
