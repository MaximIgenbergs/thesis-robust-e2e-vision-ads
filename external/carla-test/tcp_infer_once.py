#!/usr/bin/env python3
# tcp_infer_once.py — Run TCP on a single image and print outputs (no CLI).
# Uses fixed input size H=256, W=928 so cnn_feature is (8,29), matching the attention.
# Assumes repo root on PYTHONPATH or run from repo root.

import os, sys, json, warnings
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ---------------- User settings ----------------
CKPT_PATH   = os.path.expanduser("~/Downloads/best_model.ckpt")
IMAGE_PATH  = "test_image2.png"     # any RGB image; script resizes correctly
SPEED_MS    = 5.0                   # ego speed in m/s (will be divided by 12)
TARGET_XY   = (0.0, 5.0)            # target in ego frame (meters)
COMMAND     = "lanefollow"          # left,right,straight,lanefollow,changelaneleft,changelaneright or 0..5
DEVICE      = "cpu"                 # "cpu" or "cuda"
ADD_PID     = True                  # also print PID control from predicted waypoints
REPO_ROOT   = ""                    # set if not running from repo root (path containing TCP/)
# ------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

if REPO_ROOT:
    sys.path.append(os.path.abspath(REPO_ROOT))

from TCP.model import TCP
from TCP.config import GlobalConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CMD_NAMES     = ["left","right","straight","lanefollow","changelaneleft","changelaneright"]

# Fixed size derived from your measured effective stride (s_h=s_w=32): (8*32, 29*32) = (256, 928)
IMG_H, IMG_W  = 256, 928
TARGET_FEAT_HW = (8, 29)  # attention expects cnn_feature (H,W) = (8,29)

# ---------------- Helpers ----------------

def load_ckpt_into_tcp(model: nn.Module, ckpt_path: str, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise RuntimeError(f"No 'state_dict' in checkpoint. Keys: {list(ckpt.keys())}")
    sd = ckpt["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        # Lightning wrapper saved as TCP_planner.model.*
        new_sd[k[len("model."):] if k.startswith("model.") else k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if missing or unexpected:
        print(f"[warn] load_state_dict -> missing={missing}, unexpected={unexpected}")

def load_image_tensor(path: str, size_hw=(IMG_H, IMG_W)):
    img = Image.open(path).convert("RGB")
    tfm = T.Compose([
        T.Resize(size_hw),     # (H, W)
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(img).unsqueeze(0)  # (1,3,H,W)

def one_hot_command(name_or_idx):
    if isinstance(name_or_idx, int):
        idx = name_or_idx
    else:
        name = str(name_or_idx).lower()
        if name not in CMD_NAMES:
            raise SystemExit(f"Unknown command '{name}'. Use one of {CMD_NAMES} or index 0..5.")
        idx = CMD_NAMES.index(name)
    if idx < 0 or idx > 5:
        raise SystemExit("Command index must be in 0..5")
    oh = torch.zeros(6, dtype=torch.float32)
    oh[idx] = 1.0
    return oh

def summarize_tensor(t: torch.Tensor, max_list=6):
    t_cpu = t.detach().cpu()
    return {
        "shape": list(t_cpu.shape),
        "min": float(t_cpu.min()),
        "max": float(t_cpu.max()),
        "first_values": t_cpu.flatten()[:max_list].tolist()
    }

# ---------------- Main ----------------

def main():
    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")

    cfg = GlobalConfig()
    model = TCP(cfg).to(device).eval()
    load_ckpt_into_tcp(model, CKPT_PATH, strict=False)

    # Inputs
    img = load_image_tensor(IMAGE_PATH, (IMG_H, IMG_W)).to(device)  # (1,3,256,928)

    # Sanity: verify backbone feature spatial size == (8,29)
    with torch.no_grad():
        _, feat = model.perception(img)
    h_out, w_out = feat.shape[-2], feat.shape[-1]
    assert (h_out, w_out) == TARGET_FEAT_HW, f"Backbone feature {h_out,w_out} != {TARGET_FEAT_HW}"

    speed_norm = torch.tensor([[SPEED_MS/12.0]], dtype=torch.float32, device=device)  # (1,1)

    # Command one-hot (supports string or int)
    try:
        cmd_index = int(COMMAND)
        cmd_oh = one_hot_command(cmd_index)
    except (ValueError, TypeError):
        cmd_oh = one_hot_command(COMMAND)
    cmd_oh = cmd_oh.view(1, -1).to(device)  # (1,6)

    target_point = torch.tensor([list(TARGET_XY)], dtype=torch.float32, device=device)  # (1,2)
    state = torch.cat([speed_norm, target_point, cmd_oh], dim=1)  # (1, 1+2+6) = (1,9)

    # Forward
    with torch.no_grad():
        pred = model(img, state, target_point)
        mu, sigma = pred["mu_branches"], pred["sigma_branches"]      # (1,2), (1,2)
        pred_wp   = pred["pred_wp"]                                  # (1, pred_len, 2)
        pred_speed= pred["pred_speed"]                               # (1,1)

        # Policy-derived control
        throttle, steer, brake = model.get_action(mu[0], sigma[0])

        out = {
            "input_used": {"H": IMG_H, "W": IMG_W, "verified_feat_hw": [h_out, w_out]},
            "pred_speed": summarize_tensor(pred_speed),
            "mu": summarize_tensor(mu),
            "sigma": summarize_tensor(sigma),
            "pred_wp": {
                "shape": list(pred_wp.shape),
                "waypoints": pred_wp[0].detach().cpu().tolist(),
            },
            "policy_control": {
                "throttle": float(throttle.item()),
                "steer": float(steer.item()),
                "brake": float(brake.item())
            }
        }

        if ADD_PID:
            steer2, thr2, br2, meta = model.control_pid(pred_wp, speed_norm, target_point)
            out["pid_control"] = {
                "throttle": float(thr2),
                "steer": float(steer2),
                "brake": float(br2),
                "meta": meta
            }

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
