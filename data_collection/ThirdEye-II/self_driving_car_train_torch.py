import pathlib
import random
import lightning as pl
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.conf import ACCELERATOR, DEVICE, DEFAULT_DEVICE, CHECKPOINT_DIR, PROJECT_DIR, Training_Configs
from utils.conf import Track_Infos, CHECKPOINT_DIR, Training_Configs
from model.lane_keeping.vit.vit_model import ViT
import numpy as np


pl.seed_everything(42)
torch.set_float32_matmul_precision('high')
track_index = 1

def random_flip(x, y):
    if random.random() > 0.5:
        return torchvision.transforms.functional.hflip(x), -y
    return x, y


class DrivingDataset(Dataset):

    def __init__(self,
                 dataset_dir: str,
                 track_info: Track_Infos[track_index],
                 split: str = "train",
                 transform=None):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.track_info = track_info

        csv_path = self.dataset_dir.joinpath('driving_log.csv')
        column_name = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 'lap', 'sector', 'cte']

        self.metadata = pd.read_csv(csv_path, header=0)
        if list(self.metadata.columns) != column_name:
            self.metadata.columns = column_name


        self.split = split
        if self.split == "train":
            self.metadata = self.metadata[10: int(len(self.metadata) * 0.9)]
        else:
            self.metadata = self.metadata[int(len(self.metadata) * 0.9):]
        if transform == None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.AugMix(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def _load_and_preprocess(self):
        x_list = []
        y_steering_list = []
        y_throttle_list = []
        column_name = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 'lap', 'sector', 'cte']

        for drive_style in self.track_info['driving_style']:
            try:
                csv_path = self.track_info['training_data_dir'].joinpath(
                                            drive_style,
                                            'driving_log.csv')
                data_df = pd.read_csv(csv_path, header=0)

                if list(data_df.columns) != column_name:
                    data_df.columns = column_name

                if Training_Configs['AUG']['USE_LEFT_RIGHT']:
                    y_throttle_center = data_df['throttle'].values
                    y_throttle_left = y_throttle_center / 1.2
                    y_throttle_right = y_throttle_center / 1.2

                    y_center = data_df['steering'].values
                    y_left = y_center + 0.1
                    y_right = y_center - 0.1

                    new_x = np.concatenate([data_df['center'].values,
                                            data_df['left'].values,
                                            data_df['right'].values])
                    new_y_steering = np.concatenate([y_center, y_left, y_right])
                    new_y_throttle = np.concatenate([y_throttle_center, y_throttle_left, y_throttle_right])

                    x_list.append(new_x)
                    y_steering_list.append(new_y_steering)
                    y_throttle_list.append(new_y_throttle)
                else:
                    y_throttle_center = data_df['throttle'].values
                    y_steering_center = data_df['steering'].values
                    x_center = data_df['center'].values

                    x_list.append(x_center)
                    y_steering_list.append(y_steering_center)
                    y_throttle_list.append(y_throttle_center)

            except FileNotFoundError:
                print(f"Unable to read file {csv_path}")
                continue

        if not x_list:
            raise RuntimeError(
                "No driving data were provided for training. Provide correct paths to the driving_log.csv files.")

        x = np.concatenate(x_list, axis=0)
        y_steering = np.concatenate(y_steering_list, axis=0).reshape(-1, 1)
        y_throttle = np.concatenate(y_throttle_list, axis=0).reshape(-1, 1)
        y = np.concatenate((y_steering, y_throttle), axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=Training_Configs['TEST_SIZE'],
                                                            shuffle=True,
                                                            random_state=0)
        print("Loading training set completed in %s." % str(timedelta(seconds=round(duration_train))))

        print(f"Data set: {len(x)} elements")
        print(f"Training set: {len(x_train)} elements")
        print(f"Test set: {len(x_test)} elements")

        return x_train, x_test, y_train, y_test


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = Image.open(self.dataset_dir.joinpath("image", self.metadata['image_filename'].values[idx]))
        steering = self.metadata['predicted_steering_angle'].values[idx]
        steering = torch.tensor([steering], dtype=torch.float32)
        if self.split == "train":
            image, steering = random_flip(image, steering)
        return self.transform(image), steering


if __name__ == '__main__':
    # Run parameters
    input_shape = (3, 160, 320)
    max_epochs = 2000
    accelerator = ACCELERATOR
    devices = [DEVICE]
    dataset_paths = [
        'udacity_dataset_lake',
        'udacity_dataset_lake_8_8_1',
        'udacity_dataset_lake_12_8_1',
        'udacity_dataset_lake_12_12_1',
    ]

    train_dataset = torch.utils.data.ConcatDataset([
        DrivingDataset(dataset_dir=PROJECT_DIR.joinpath(dataset, "lake_sunny_day"), split="train")
        for dataset in dataset_paths
    ])
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        prefetch_factor=4,
        num_workers=16,
    )

    val_dataset = torch.utils.data.ConcatDataset([
        DrivingDataset(dataset_dir=PROJECT_DIR.joinpath(dataset, "lake_sunny_day"), split="val", transform=torchvision.transforms.ToTensor())
        for dataset in dataset_paths
    ])
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        prefetch_factor=2,
        num_workers=8,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR.joinpath("lane_keeping", "vit"),
        filename="vit",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    earlystopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=20)
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, earlystopping_callback],
        devices=devices,
    )

    driving_model = ViT()
    trainer.fit(
        driving_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
