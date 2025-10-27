# File: models/dave2_legacy/train.py

import sys
import pathlib

# Ensure project root is on PYTHONPATH for shared utils
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
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

from models.dave2_v1.model import Dave2
from utils.device_config import ACCELERATOR, DEVICE
from utils.paths import get_model_dir, get_fig_dir, TRAIN_DATA_DIR

# Reproducibility
pl.seed_everything(42)
torch.set_float32_matmul_precision('high')


def random_flip(image, steering):
    """
    Randomly flip the image horizontally and negate the steering angle
    with 50% probability.
    """
    if random.random() > 0.5:
        flipped = torchvision.transforms.functional.hflip(image)
        return flipped, -steering
    return image, steering


class DrivingDataset(Dataset):
    """
    A PyTorch dataset for single-frame driving data.
    Splits the log.csv into 90% train / 10% validation after skipping
    the first 10 frames.
    """
    def __init__(self, dataset_dir: str, split: str = "train", transform=None):
        self.dataset_dir = pathlib.Path(dataset_dir)
        df = pd.read_csv(self.dataset_dir / "log.csv")
        total = len(df)
        split_idx = int(total * 0.9)
        if split == "train":
            subset = df.iloc[10:split_idx]
        else:
            subset = df.iloc[split_idx:]
        self.metadata = subset.reset_index(drop=True)

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
    def get_image(self, idx):
        """
        Load and return a PIL image for a given index.
        """
        filename = self.metadata.at[idx, "image_filename"]
        return Image.open(self.dataset_dir / "image" / filename)

    def __getitem__(self, idx):
        """
        Returns (image_tensor, steering_tensor) with random flip augmentation.
        """
        img = self.get_image(idx)
        steer_val = float(self.metadata.at[idx, "predicted_steering_angle"])
        steering = torch.tensor([steer_val], dtype=torch.float32)
        img, steering = random_flip(img, steering)
        return self.transform(img), steering


if __name__ == "__main__":
    # Hyperparameters and device setup
    input_shape = (3, 160, 320)
    max_epochs = 2000
    accelerator = ACCELERATOR
    devices = [DEVICE]

    # Data loaders
    dataset_dir = TRAIN_DATA_DIR
    train_dataset = DrivingDataset(dataset_dir, split="train")
    val_dataset = DrivingDataset(dataset_dir, split="val",
                                 transform=torchvision.transforms.ToTensor())

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

    # Callbacks and logging
    checkpoint_dir = get_model_dir("dave2_v1")
    figure_dir = get_fig_dir("dave2_v1")
    csv_logger = CSVLogger(save_dir=str(figure_dir), name="dave2_v1")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="dave2_v1",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=20,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=csv_logger,
    )

    # Training
    model = Dave2()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final checkpoint
    final_ckpt = checkpoint_dir / "final_dave2_v1.ckpt"
    trainer.save_checkpoint(str(final_ckpt))
    print(f"Training complete. Model saved to: {final_ckpt}")

    # Plot and save loss curves
    metrics_file = pathlib.Path(csv_logger.log_dir) / "metrics.csv"
    metrics = pd.read_csv(metrics_file).groupby("epoch").last().reset_index()

    plt.figure()
    plt.plot(metrics["epoch"], metrics["train/loss"], label="train_loss")
    plt.plot(metrics["epoch"], metrics["val/loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    plot_name = f"loss_curve_{timestamp}.png"
    plt.savefig(str(figure_dir / plot_name))
    plt.close()
    print(f"Loss curve saved to: {figure_dir/plot_name}")
