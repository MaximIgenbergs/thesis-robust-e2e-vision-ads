# sims/udacity/models/vit/model.py

from __future__ import annotations
from pathlib import Path
import sys

import torch
from torch import nn
import pytorch_lightning as pl
import timm

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sims.udacity.models.vit.config import (
    BACKBONE_NAME,
    LEARNING_RATE,
    ALPHA_STEER,
)


class ViTDriving(pl.LightningModule):
    """
    Representative ViT-B/16 model:

    - Backbone: timm `vit_base_patch16_224.augreg_in21k_ft_in1k`
    - Head: small MLP for [steering, throttle] regression.
    - Loss: weighted MSE (steer vs throttle).
    """

    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        lr: float = LEARNING_RATE,
        alpha_steer: float = ALPHA_STEER,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Pretrained ViT-B/16
        self.backbone = timm.create_model(backbone_name, pretrained=True)

        # Remove classifier, keep feature extractor
        if hasattr(self.backbone, "reset_classifier"):
            in_features = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0)  # makes forward(x) return features
        else:
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        # Regression head: [steer, throttle]
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        """
        x: (B, 3, H, W)

        After reset_classifier(0), timm's forward(x) returns pooled features (B, D).
        """
        feats = self.backbone(x)   # (B, D), NOT tokens
        out = self.head(feats)     # (B, 2)
        return out

    def _loss(self, preds, labels):
        """
        preds:  (B, 2)
        labels: (B, 2) ideally; handle minor shape issues defensively.
        """
        labels = labels.to(preds.device).float()

        # Defensive shape fixes in case something weird slips through
        if labels.ndim == 1:
            labels = torch.stack([labels, torch.zeros_like(labels)], dim=-1)
        elif labels.ndim == 2 and labels.shape[1] == 1 and preds.shape[1] == 2:
            labels = torch.cat([labels, torch.zeros_like(labels)], dim=-1)

        # Now both preds and labels should be (B, 2)
        diff = preds - labels
        s_loss = (diff[:, 0] ** 2).mean()
        t_loss = (diff[:, 1] ** 2).mean()
        alpha = self.hparams.alpha_steer
        return alpha * s_loss + (1.0 - alpha) * t_loss

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self._loss(preds, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self._loss(preds, labels)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        return optimizer
