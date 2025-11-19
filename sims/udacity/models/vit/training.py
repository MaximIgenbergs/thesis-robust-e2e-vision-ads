import json
import pathlib
import random
import sys
from pathlib import Path
from datetime import datetime

import lightning as pl
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader

# add project root to path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sims.udacity.logging.training_runs import make_run_dir, write_meta, loss_plot
from sims.udacity.models.vit.config import DATA_DIR, MAP_NAME, MODEL_NAME, EPOCHS, BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, PATIENCE, ACCELERATOR, DEVICE
from sims.udacity.models.simple_vit.model import ViT

pl.seed_everything(42)
torch.set_float32_matmul_precision("high")


def random_flip(x, y):
    if random.random() > 0.5:
        return torchvision.transforms.functional.hflip(x), -y
    return x, y


class DrivingDataset(Dataset):
    def __init__(self, dataset_dir: str | pathlib.Path, split: str = "train", transform=None):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.split = split

        # all record_*.json files form the index
        all_records = sorted(self.dataset_dir.glob("record_*.json"))
        if not all_records:
            raise RuntimeError(f"No record_*.json files found in {self.dataset_dir}")

        n = len(all_records)

        # simple temporal split, like before: skip first 10 entries, then 90/10 train/val
        train_start = min(10, n)
        val_start = int(n * 0.9)

        if split == "train":
            self.indices = list(range(train_start, val_start))
        else:
            self.indices = list(range(val_start, n))

        self.records = all_records

        if not self.indices:
            raise RuntimeError(f"Split '{split}' is empty in {self.dataset_dir} (n={n})")

        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.AugMix(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        rec_path = self.records[real_idx]

        # matching image_*.jpg for record_*.json
        num_part = rec_path.stem.split("_", 1)[1]  # "record_000000" -> "000000"
        image_path = self.dataset_dir / f"image_{num_part}.jpg"

        with open(rec_path, "r") as f:
            record = json.load(f)

        # steering from JSON (string -> float)
        steering = float(record["user/angle"])
        steering = torch.tensor([steering], dtype=torch.float32)

        image = Image.open(image_path).convert("RGB")

        if self.split == "train":
            image, steering = random_flip(image, steering)

        return self.transform(image), steering


class EpochHistory(pl.Callback):
    """Collect per-epoch train/val loss in Keras-style keys for loss_plot."""

    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = int(trainer.current_epoch)

        row = {"epoch": epoch}

        # try several common Lightning naming patterns
        train_keys = ["train/loss", "train_loss", "loss"]
        val_keys = ["val/loss", "val_loss"]

        for src_keys, dst in ((train_keys, "loss"), (val_keys, "val_loss")):
            for k in src_keys:
                if k in metrics:
                    v = metrics[k]
                    if isinstance(v, torch.Tensor):
                        row[dst] = float(v.detach().cpu().item())
                    else:
                        row[dst] = float(v)
                    break

        self.records.append(row)


if __name__ == "__main__":
    input_shape = (3, 160, 320)

    accelerator = ACCELERATOR
    devices = [DEVICE]

    dataset_dir = pathlib.Path(DATA_DIR)

    train_dataset = DrivingDataset(dataset_dir=dataset_dir, split="train")
    val_dataset = DrivingDataset(
        dataset_dir=dataset_dir,
        split="val",
        transform=torchvision.transforms.ToTensor(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        prefetch_factor=4,
        num_workers=16,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        prefetch_factor=2,
        num_workers=8,
    )

    # logging setup in "DAVE2-GRU style"
    run_dir = make_run_dir(model_key=MODEL_NAME, map_name=MAP_NAME)
    best_path = run_dir / "best_model.ckpt"
    hist_csv = run_dir / "history.csv"

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_model",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    earlystopping_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=PATIENCE,
    )
    history_callback = EpochHistory()

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, earlystopping_callback, history_callback],
        devices=devices,
    )

    driving_model = ViT()

    trainer.fit(
        driving_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # post-training: write history, meta, and loss plot
    if history_callback.records:
        df = pd.DataFrame(history_callback.records)
        df.to_csv(hist_csv, index=False)

        # Keras-like history dict for loss_plot
        history_like = type(
            "History",
            (),
            {"history": {k: df[k].tolist() for k in df.columns if k in ("loss", "val_loss")}},
        )()

        loss_png = loss_plot(history_like, run_dir, title=f"ViT on {MAP_NAME}")
    else:
        loss_png = None

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    meta = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "framework": "pytorch-lightning",
        "model": MODEL_NAME,
        "map": MAP_NAME,
        "checkpoint_path": str(checkpoint_callback.best_model_path or best_path),
        "input_shape": list(input_shape),
        "seed": 42,
        "val_split": 0.1,  # effective split from indexing
        "optimizer": {
            "type": "adam",  # lr is inside ViT; add here if you want
        },
        "batch_size_train": BATCH_SIZE_TRAIN,
        "batch_size_val": BATCH_SIZE_VAL,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "augmentations": ["AugMix", "horizontal_flip"],
        "data": {
            "data_dir": str(DATA_DIR),
        },
        "counts": {
            "train_samples": int(n_train),
            "val_samples": int(n_val),
        },
    }

    write_meta(run_dir, meta)

    print(f"[train:{MODEL_NAME}] Best model: {checkpoint_callback.best_model_path or best_path}")
    if loss_png is not None:
        print(f"[train:{MODEL_NAME}] Loss curve: {loss_png}")
    print(f"[train:{MODEL_NAME}] History CSV: {hist_csv}")
    print(f"[train:{MODEL_NAME}] Meta: {run_dir / 'meta.json'}")
