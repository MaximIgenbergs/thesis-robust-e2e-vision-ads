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

# Ensure reproducibility
pl.seed_everything(42)
torch.set_float32_matmul_precision('high')


def random_flip(image, steering):
    """
    Randomly flips the image horizontally and negates the steering angle
    with 50% probability.
    """
    if random.random() > 0.5:
        flipped = torchvision.transforms.functional.hflip(image)
        return flipped, -steering
    return image, steering


class DrivingDataset(Dataset):
    """
    PyTorch dataset for vision-based driving data.
    Splits the log.csv into training and validation subsets.
    """
    def __init__(self, dataset_dir: str, split: str = "train", transform=None):
        self.dataset_dir = pathlib.Path(dataset_dir)
        all_data = pd.read_csv(self.dataset_dir / "log.csv")
        total = len(all_data)
        split_idx = int(total * 0.9)
        if split == "train":
            self.metadata = all_data.iloc[10:split_idx].reset_index(drop=True)
        else:
            self.metadata = all_data.iloc[split_idx:].reset_index(drop=True)

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
    def get_image(self, index):
        """
        Loads and returns a PIL image for a given index.
        """
        filename = self.metadata.at[index, "image_filename"]
        return Image.open(self.dataset_dir / "image" / filename)

    def __getitem__(self, index):
        """
        Returns a (image_tensor, steering_tensor) pair.
        Applies random horizontal flip augmentation during training.
        """
        img = self.get_image(index)
        steer_value = float(self.metadata.at[index, "predicted_steering_angle"])
        steering = torch.tensor([steer_value], dtype=torch.float32)
        if self.metadata is not None and index < len(self.metadata):
            img, steering = random_flip(img, steering)
        return self.transform(img), steering


if __name__ == "__main__":
    # Configure training parameters
    input_shape = (3, 160, 320)
    max_epochs = 2000
    devices = [DEVICE]
    accelerator = ACCELERATOR

    # Prepare data loaders
    dataset_dir = TRAIN_DATA_DIR
    train_dataset = DrivingDataset(dataset_dir=dataset_dir, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=16,
        prefetch_factor=4,
    )
    val_dataset = DrivingDataset(
        dataset_dir=dataset_dir,
        split="val",
        transform=torchvision.transforms.ToTensor(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        prefetch_factor=2,
    )

    # Configure checkpointing, early stopping, and logging
    checkpoint_dir = get_model_dir("vit")
    figure_dir = get_fig_dir("vit")
    csv_logger = CSVLogger(save_dir=str(figure_dir), name="vit")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="vit",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=20,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=csv_logger,
    )

    # Instantiate and train the model
    model = ViT(input_shape=input_shape)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save the final checkpoint
    final_checkpoint = checkpoint_dir / "final_vit.ckpt"
    trainer.save_checkpoint(str(final_checkpoint))
    print(f"Training complete. Model saved to: {final_checkpoint}")

    # Load logged metrics and plot loss curves
    metrics_path = pathlib.Path(csv_logger.log_dir) / "metrics.csv"
    metrics_df = pd.read_csv(metrics_path)
    epoch_metrics = metrics_df.groupby("epoch").last().reset_index()

    plt.figure()
    plt.plot(epoch_metrics["epoch"], epoch_metrics["train/loss"], label="train_loss")
    plt.plot(epoch_metrics["epoch"], epoch_metrics["val/loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    plot_file = f"loss_curve_{ts}.png"
    plt.savefig(str(figure_dir / plot_file))
    plt.close()
    print(f"Loss curve saved to: {figure_dir/plot_file}")
