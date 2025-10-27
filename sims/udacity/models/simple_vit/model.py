# models/vit/model.py

import torch
from torch import nn
import pytorch_lightning as pl

class PatchEmbedding(nn.Module):
    """
    Same as before: Conv‐based patch split + [CLS] token + positional embed.
    """
    def __init__(self, image_size=(160, 320), patch_size=16, in_channels=3, embed_dim=128, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)                            # (B, E, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)            # (B, N, E)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)              # (B, 1+N, E)
        x = x + self.pos_embed
        return self.dropout(x)

class ViTModule(nn.Module):
    """
    Transformer encoder stack unchanged.
    """
    def __init__(self, embed_dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=mlp_dim, dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)          # (seq_len, B, E)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)          # (B, seq_len, E)
        return self.norm(x)

class ViT(pl.LightningModule):
    """
    Vision Transformer for steering+throttle regression, with weighted loss.
    """
    def __init__(self,
                 image_size=(160, 320), patch_size=16, in_channels=3,
                 embed_dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim, dropout)
        self.vit         = ViTModule(embed_dim, depth, heads, mlp_dim, dropout)
        self.mlp_head    = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.vit(x)
        cls = x[:, 0]                # (B, E)
        return self.mlp_head(cls)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        # weighted MSE: steer vs throttle
        from models.utils.training_defaults import ALPHA_STEER
        s_loss = nn.functional.mse_loss(preds[:,0], labels[:,0])
        t_loss = nn.functional.mse_loss(preds[:,1], labels[:,1])
        loss = ALPHA_STEER * s_loss + (1 - ALPHA_STEER) * t_loss
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        from models.utils.training_defaults import ALPHA_STEER
        s_loss = nn.functional.mse_loss(preds[:,0], labels[:,0])
        t_loss = nn.functional.mse_loss(preds[:,1], labels[:,1])
        loss = ALPHA_STEER * s_loss + (1 - ALPHA_STEER) * t_loss
        self.log('val/loss', loss, prog_bar=True)

    def configure_optimizers(self):
        from models.utils.training_defaults import LEARNING_RATE
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
