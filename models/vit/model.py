import math
import torchvision.transforms.functional as TF
from typing import Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import VisionTransformer

from models.utils.training_defaults import ALPHA_STEER


class ViT(pl.LightningModule):
    """
    Vision Transformer that predicts both steering and throttle.
    Uses a weighted MSE loss between the two outputs.
    """

    def __init__(
        self,
        input_shape: Tuple[int,int,int] = (3,160,320),
        learning_rate: float = 2e-4,
        alpha_steer: float = ALPHA_STEER,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.alpha_steer   = alpha_steer

        # Two-output regression: steering + throttle
        self.model = VisionTransformer(
            image_size=160,
            patch_size=8,
            num_classes=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=512,
            mlp_dim=128
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 3, 160, 320)
        # Resize to (B, 3, 160, 160) using interpolate (no AA)
        x_resized = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        return self.model(x_resized)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch                      # targets: [B,2]
        preds = self(imgs)                         # preds:   [B,2]
        steer_p, thr_p   = preds[:,0], preds[:,1]
        steer_t, thr_t   = targets[:,0], targets[:,1]

        loss_s = F.mse_loss(steer_p, steer_t)
        loss_t = F.mse_loss(thr_p, thr_t)
        loss   = self.alpha_steer * loss_s + (1-self.alpha_steer)*loss_t

        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log("train/rmse", math.sqrt(loss), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        steer_p, thr_p = preds[:,0], preds[:,1]
        steer_t, thr_t = targets[:,0], targets[:,1]

        loss_s = F.mse_loss(steer_p, steer_t)
        loss_t = F.mse_loss(thr_p, thr_t)
        loss   = self.alpha_steer * loss_s + (1-self.alpha_steer)*loss_t

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/rmse", math.sqrt(loss), prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        steer_p, thr_p = preds[:,0], preds[:,1]
        steer_t, thr_t = targets[:,0], targets[:,1]

        loss_s = F.mse_loss(steer_p, steer_t)
        loss_t = F.mse_loss(thr_p, thr_t)
        loss   = self.alpha_steer * loss_s + (1-self.alpha_steer)*loss_t

        self.log("test/loss", loss, prog_bar=True)
        self.log("test/rmse", math.sqrt(loss), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
