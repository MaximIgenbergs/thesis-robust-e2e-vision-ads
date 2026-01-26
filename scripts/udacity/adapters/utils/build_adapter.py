
from pathlib import Path

from scripts import abs_path, CKPTS_DIR
from scripts.udacity.adapters.dave2_adapter import Dave2Adapter
from scripts.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter
from scripts.udacity.adapters.vit_adapter import ViTAdapter

def build_adapter(model_name: str, model_cfg: dict):
    ckpt_rel = model_cfg.get("checkpoint")
    ckpt = abs_path(CKPTS_DIR / ckpt_rel) if ckpt_rel else None

    image_size_hw = tuple(model_cfg.get("image_size_hw", [240, 320]))
    seq_len = model_cfg.get("sequence_length", 3)

    if model_name == "dave2":
        return (Dave2Adapter(weights=ckpt, image_size_hw=image_size_hw, device=None), ckpt)
    if model_name == "dave2_gru":
        return (Dave2GRUAdapter(weights=ckpt, image_size_hw=image_size_hw, seq_len=seq_len, device=None), ckpt)
    if model_name == "vit":
        return (ViTAdapter(weights=ckpt, image_size_hw=image_size_hw, device=None), ckpt)

    raise ValueError(f"Model '{model_name}' not defined.")