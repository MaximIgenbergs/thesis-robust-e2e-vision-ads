import math
import torchvision.utils
from typing import Tuple
import lightning as pl
import torch
from torch import Tensor
from torchvision.models import VisionTransformer


class ViT(pl.LightningModule):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 160, 320),
                 learning_rate: float = 2e-4,
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        # self.example_input_array = torch.zeros(size=self.input_shape)
        self.model = VisionTransformer(image_size=160, patch_size=8, num_classes=1, num_layers=2, num_heads=2, hidden_dim=512, mlp_dim=128)
        self.loss = torch.nn.MSELoss()

    def forward(self, x: Tensor):
        # if len(x.shape) == 4:
        #     x = x.unsqueeze(0)
        # print(x.shape)
        # print(torchvision.transforms.functional.resize(x, (160,160)).unsqueeze(0).shape)
        # return self.model(torchvision.transforms.functional.resize(x, (160,160)).unsqueeze(0)).squeeze(0)
        return self.model(torchvision.transforms.functional.resize(x, (160,160)))

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
