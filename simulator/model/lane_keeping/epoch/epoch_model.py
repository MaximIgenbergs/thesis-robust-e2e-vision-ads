import math
from typing import Tuple
import lightning as pl
import torch
from torch import Tensor


class Epoch(pl.LightningModule):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 160, 320),
                 learning_rate: float = 2e-4,
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.example_input_array = torch.zeros(size=self.input_shape)
        flat_shape = int(128 * math.floor(input_shape[-2] / 64) * math.floor(input_shape[-1] / 64))
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Dropout(p=0.25),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Dropout(p=0.25),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Dropout(p=0.5),
            torch.nn.Flatten(start_dim=-3, end_dim=-1),
            torch.nn.Linear(in_features=flat_shape, out_features=1024),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=1)
        )
        self.loss = torch.nn.MSELoss()

    def forward(self, x: Tensor):
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int = 0):
        img, true = batch
        pred = self.forward(x=img)
        loss = self.loss(pred, true)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log("train/rmse", math.sqrt(loss), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        img, true = batch
        pred = self(img)
        loss = self.loss(pred, true)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/rmse", math.sqrt(loss), prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        img, true = batch
        pred = self(img)
        loss = self.loss(pred, true)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/rmse", math.sqrt(loss), prog_bar=True)
        return loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        img, _ = batch
        pred = self(img)
        return pred

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]