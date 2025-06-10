import sys
import pathlib

# Add project root to PYTHONPATH so shared utils can be imported
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import random
from functools import cache
import datetime

import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
import matplotlib.pyplot as plt

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader

from model import ViT
from models.utils.device_config import ACCELERATOR, DEVICE
from models.utils.paths import get_model_dir, get_fig_dir, TRAIN_DATA_DIR
from models.utils.training_defaults import ALPHA_STEER

# Ensure reproducibility
pl.seed_everything(42)
torch.set_float32_matmul_precision('high')


def random_flip(image: Image.Image, target: torch.Tensor):
    """
    Randomly flip the image horizontally and negate only the steering channel.
    `target` is a 2-element tensor [steering, throttle].
    """
    if random.random() > 0.5:
        flipped = torchvision.transforms.functional.hflip(image)
        steer, thr = target.unbind(0)
        return flipped, torch.tensor([-steer, thr], dtype=torch.float32)
    return image, target


class DrivingDataset(Dataset):
    """
    PyTorch dataset for vision‐based driving data.
    Splits `log.csv` into 90% train / 10% val after skipping the first 10 frames.
    Expects `log.csv` to have columns:
      - image_filename
      - predicted_steering_angle
      - predicted_throttle
    """
    def __init__(self, dataset_dir: str, split: str = "train", transform=None):
        self.dataset_dir = pathlib.Path(dataset_dir)
        df = pd.read_csv(self.dataset_dir / "log.csv")
        n_total = len(df)
        split_idx = int(n_total * 0.9)

        if split == "train":
            self.metadata = df.iloc[10:split_idx].reset_index(drop=True)
        else:
            self.metadata = df.iloc[split_idx:].reset_index(drop=True)

        self.transform = (
            transform
            if transform is not None
            else torchvision.transforms.Compose([
                torchvision.transforms.AugMix(),
                torchvision.transforms.ToTensor(),
            ])
        )

    def __len__(self):
        return len(self.metadata)

    @cache
    def get_image(self, idx: int) -> Image.Image:
        filename = self.metadata.at[idx, "image_filename"]
        return Image.open(self.dataset_dir / "image" / filename)

    def __getitem__(self, idx: int):
        img = self.get_image(idx)

        # build 2-element target: [steering, throttle]
        steer = float(self.metadata.at[idx, "predicted_steering_angle"])
        thr   = float(self.metadata.at[idx, "predicted_throttle"])
        target = torch.tensor([steer, thr], dtype=torch.float32)

        # apply flip‐only augmentation during training
        if idx < len(self.metadata):  # split logic already applied
            img, target = random_flip(img, target)

        return self.transform(img), target


if __name__ == "__main__":
    # Training configuration
    input_shape = (3, 160, 320)
    max_epochs  = 2000
    devices     = [DEVICE]
    accelerator = ACCELERATOR

    # Data loaders
    dataset_dir   = TRAIN_DATA_DIR
    train_ds      = DrivingDataset(dataset_dir, split="train")
    val_ds        = DrivingDataset(dataset_dir, split="val", transform=torchvision.transforms.ToTensor())
    train_loader  = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=8, persistent_workers=True, prefetch_factor=4)
    val_loader    = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=8, persistent_workers=True, prefetch_factor=2)

    # Checkpointing, early stop, CSV logging
    ckpt_dir = get_model_dir("vit")
    fig_dir  = get_fig_dir("vit")
    csv_logger = CSVLogger(save_dir=str(fig_dir), name="vit")

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_model",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    earlystop_cb = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=20,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=[checkpoint_cb, earlystop_cb],
        logger=csv_logger,
    )

    # Fit the model
    model = ViT(input_shape=input_shape, learning_rate=2e-4, alpha_steer=ALPHA_STEER)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save final checkpoint
    final_ckpt = ckpt_dir / "final_model.ckpt"
    trainer.save_checkpoint(str(final_ckpt))
    print(f"Training complete. Model saved to: {final_ckpt}")

    # Plot & save loss curve
    metrics_path  = pathlib.Path(csv_logger.log_dir) / "metrics.csv"
    metrics_df    = pd.read_csv(metrics_path).groupby("epoch").last().reset_index()

    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["train/loss"], label="train_loss")
    plt.plot(metrics_df["epoch"], metrics_df["val/loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    plot_name = f"loss_curve_{ts}.png"
    plt.savefig(str(fig_dir / plot_name))
    plt.close()
    print(f"Loss curve saved to: {fig_dir/plot_name}")
