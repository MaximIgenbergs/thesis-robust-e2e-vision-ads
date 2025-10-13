#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_tcp.py — Minimal single-image forward test for a TCP-style model.
- Reads TCP settings from config.yaml -> section 'tcp' (see schema below).
- Loads Lightning (.ckpt) or TorchScript (.pt).
- Runs one screenshot through the model and prints the raw output.
- No CLI, paste your screenshot path into IMAGE_PATH below.

Expected config.yaml snippet:

tcp:
  ckpt: "~/Downloads/best_model.ckpt"     # path to your .ckpt or .pt
  format: "lightning"                     # "lightning" | "torchscript"
  device: "cuda:0"                        # or "cpu"
  input_w: 256
  input_h: 256
  n_cmds: 9                               # override if auto-infer fails
  aim_dist: 4.0                           # meters (unused here)
  model:
    class: "TCP.model.TCP"
    kwargs: {}
    config_yaml: null
    config_class: "TCP.config:GlobalConfig"
    config_override: {}
"""

from __future__ import annotations

import os, json, importlib, inspect
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict

import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F  # noqa: F401
import torch.nn as nn

# ======= Paste your screenshot absolute path here =======
IMAGE_PATH = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/external/carla-test/test_image.png"
# Optional: set a target point in ego frame (meters) if your model needs it; else leave None
TARGET_POINT_XY: Optional[Tuple[float,float]] = None
# Optional: command and speed fed to the model
CMD_STR = "STRAIGHT"
SPEED_MPS = 0.0
# Path to your YAML with the 'tcp' section:
CONFIG_PATH = "config.yaml"
# ========================================================

CMD_ORDER = ["FOLLOW", "LEFT", "RIGHT", "STRAIGHT"]

def encode_command(cmd: str, dim: int = 4) -> torch.Tensor:
    c = cmd.upper()
    key = "FOLLOW"
    if "LEFT" in c: key = "LEFT"
    elif "RIGHT" in c: key = "RIGHT"
    elif "STRAIGHT" in c or "LANEFOLLOW" in c: key = "STRAIGHT" if "STRAIGHT" in c else "FOLLOW"
    onehot = torch.zeros(1, dim, dtype=torch.float32)
    idx = CMD_ORDER.index(key) if key in CMD_ORDER else 0
    if idx < dim: onehot[0, idx] = 1.0
    return onehot

def encode_command3(cmd: str) -> torch.Tensor:
    c = cmd.upper()
    idx = 2
    if "LEFT" in c: idx = 0
    elif "RIGHT" in c: idx = 1
    v = torch.zeros(1, 3, dtype=torch.float32); v[0, idx] = 1.0
    return v

def load_image_tensor(path: str, size_wh: Tuple[int,int]) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize(size_wh, Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).float() / 255.0  # H,W,3
    return x.permute(2,0,1).unsqueeze(0)  # 1,3,H,W

def infer_cmd_dim_from_params(model: torch.nn.Module) -> int:
    cands = set()
    for _, p in getattr(model, "named_parameters", lambda: [])():
        if getattr(p, "ndim", 0) == 2:
            d = p.shape[1]
            if d in (3,4,5,6,8,9,10): cands.add(d)
    for pref in (9,8,6,5,4,3,10):
        if pref in cands: return pref
    return 4

def build_model_from_tcp(tcp: Dict[str, Any]) -> Tuple[Any, torch.device, int]:
    device = torch.device(tcp.get("device", "cpu"))
    fmt = str(tcp["format"]).lower()
    ckpt = os.path.expanduser(tcp["ckpt"])
    if not os.path.isfile(ckpt): raise FileNotFoundError(f"ckpt not found: {ckpt}")

    model = None
    cmd_dim = int(tcp.get("n_cmds")) if tcp.get("n_cmds") is not None else None

    if fmt == "torchscript":
        model = torch.jit.load(ckpt, map_location=device).eval()

    elif fmt == "lightning":
        mconf = tcp.get("model", {})
        cls_path = mconf.get("class")
        if not cls_path: raise ValueError("tcp.model.class must be set for format=lightning")
        module, cls_name = cls_path.rsplit(".", 1)
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        kwargs = dict(mconf.get("kwargs") or {})

        # Optionally provide a config object
        if mconf.get("config_yaml"):
            with open(mconf["config_yaml"], "r") as f:
                cfg_dict = yaml.safe_load(f)
            kwargs.setdefault("config", cfg_dict)
        if mconf.get("config_class"):
            mod_path, cfg_cls_name = mconf["config_class"].split(":", 1)
            mod_cfg = importlib.import_module(mod_path)
            cfg_cls = getattr(mod_cfg, cfg_cls_name)
            overrides = dict(mconf.get("config_override") or {})
            cfg_obj = cfg_cls(**overrides)
            kwargs["config"] = cfg_obj

        def _safe_load(path: str):
            try:
                return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
            except Exception:
                print("[test_tcp] Warning: falling back to torch.load(weights_only=False). Use only trusted checkpoints.")
                return torch.load(path, map_location="cpu")

        if hasattr(cls, "load_from_checkpoint"):
            try:
                model = cls.load_from_checkpoint(ckpt, map_location=device, **kwargs)
            except ModuleNotFoundError:
                obj = _safe_load(ckpt); state = obj.get("state_dict", obj)
                m = cls(**kwargs); m.load_state_dict(state, strict=False); model = m
        else:
            obj = _safe_load(ckpt); state = obj.get("state_dict", obj)
            # strip common prefixes
            if isinstance(state, dict):
                fixed = {}
                for k,v in state.items():
                    nk = k
                    for pref in ("model.","net.","module."):
                        if nk.startswith(pref): nk = nk[len(pref):]
                    fixed[nk] = v
                state = fixed
            m = cls(**kwargs); m.load_state_dict(state, strict=False); model = m

        model.to(device).eval()
        if cmd_dim is None:
            cmd_dim = infer_cmd_dim_from_params(model)

    else:
        raise ValueError("tcp.format must be 'lightning' or 'torchscript'")

    if cmd_dim is None: cmd_dim = 4
    return model, device, cmd_dim



def _make_initial_states(model, device, batch_size=1):
    """
    Build a list of plausible initial 'state' objects, trying model-provided
    initializers first, then shapes inferred from LSTM/GRU submodules.
    Returns a list ordered from most to least likely to work.
    """
    cands = []

    # 1) Try common model-provided initializers
    init_names = ["init_state", "initialize_state", "get_initial_state",
                  "get_init_state", "reset_state"]
    for name in init_names:
        fn = getattr(model, name, None)
        if callable(fn):
            # Try a few signature variants
            for args in [
                (batch_size, device),
                (batch_size,),
                (),
            ]:
                try:
                    st = fn(*args)
                    cands.append(st)
                except Exception:
                    pass  # keep trying

    # 2) Infer from LSTM/GRU modules
    lstm_gru_specs = []
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU)):
            num_layers = getattr(m, "num_layers", 1)
            hidden_size = getattr(m, "hidden_size", None)
            bidir = bool(getattr(m, "bidirectional", False))
            if hidden_size is None:
                continue
            num_dirs = 2 if bidir else 1
            lstm_gru_specs.append((type(m), num_layers, num_dirs, hidden_size))
            # only first recurrent block is usually enough
            break

    for mtype, L, D, H in lstm_gru_specs:
        # Standard RNN state shapes: [L*D, B, H]
        shape = (L * D, batch_size, H)
        h0 = torch.zeros(shape, dtype=torch.float32, device=device)
        if mtype is nn.LSTM:
            c0 = torch.zeros(shape, dtype=torch.float32, device=device)
            cands.append((h0, c0))     # (h,c) tuple
            cands.append({"h": h0, "c": c0})
        # Many repos also accept a flat [B,H] projection
        cands.append(torch.zeros(batch_size, H, dtype=torch.float32, device=device))
        # Single-vector fallbacks
        cands.append(torch.zeros(1, H, dtype=torch.float32, device=device))

    # 3) Very generic fallbacks
    cands += [
        None,                         # some models accept None for first step
        {},                           # or an empty dict
        torch.zeros(1, 1, device=device),
        torch.zeros(1, 4, device=device),
    ]
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for st in cands:
        key = (type(st), tuple(st[0].shape) if isinstance(st, tuple) and len(st)>0 and torch.is_tensor(st[0])
               else tuple(st.shape) if torch.is_tensor(st)
               else tuple(sorted(st.keys())) if isinstance(st, dict)
               else id(st))
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(st)
    return uniq

import torch.nn as nn
import itertools

def _state_probe_grid(device, batch_size=1):
    # Common hidden sizes & layer counts to try
    hidden_sizes = [32, 64, 128, 256, 512]
    layers = [1, 2]
    bidirs = [1, 2]  # 1=uni, 2=bi (D)
    cands = []

    # (h,c) tuple for LSTM + dict {"h":h,"c":c}
    for H, L, D in itertools.product(hidden_sizes, layers, bidirs):
        shape = (L * D, batch_size, H)
        h0 = torch.zeros(shape, dtype=torch.float32, device=device)
        c0 = torch.zeros(shape, dtype=torch.float32, device=device)
        cands += [(h0, c0), {"h": h0, "c": c0}]

    # GRU/RNN-like: [L*D, B, H] tensor; also flattened [B,H] and [1,H]
    for H, L, D in itertools.product(hidden_sizes, layers, bidirs):
        shape = (L * D, batch_size, H)
        cands += [
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(batch_size, H, dtype=torch.float32, device=device),
            torch.zeros(1, H, dtype=torch.float32, device=device),
        ]

    # Very generic fallbacks
    cands += [None, {}, torch.zeros(1, 1, device=device)]
    # De-dup (by type+shape signature)
    seen, uniq = set(), []
    for st in cands:
        key = (
            "tuple", tuple(st[0].shape) if isinstance(st, tuple) else None
        ) if isinstance(st, tuple) else (
            "dict", tuple(sorted(st.keys())) if isinstance(st, dict) else None
        ) if isinstance(st, dict) else (
            "tensor", tuple(st.shape) if torch.is_tensor(st) else None
        ) if torch.is_tensor(st) else ("none", None)
        if key in seen: continue
        seen.add(key); uniq.append(st)
    return uniq

@torch.no_grad()
def forward_one(model, image, cmd_str, speed, tp_xy, device, cmd_dim):
    """
    Special-cased for signature (img, state, target_point).
    - Ensures target_point is provided
    - Infers the required state dim H by probing and parsing Linear matmul error
    - Returns raw model output
    """
    # Preprocess
    x = image.to(device)                             # [1,3,H,W]
    if tp_xy is None:
        raise RuntimeError("Model requires target_point but none was provided.")
    tp = torch.tensor([[float(tp_xy[0]), float(tp_xy[1])]], dtype=torch.float32, device=device)

    # Helper: try a given H, return output or raise with message
    def try_with_H(H: int):
        st = torch.zeros(1, H, dtype=torch.float32, device=device)     # state is [1, H]
        return model(img=x, state=st, target_point=tp)

    # First, try a few common dims quickly
    candidates = [32, 64, 128, 256, 512, 1024]
    first_error = None
    for H in candidates:
        try:
            out = try_with_H(H)
            print(f"[test_tcp] used state dim H={H}")
            return out
        except Exception as e:
            msg = str(e)
            if first_error is None:
                first_error = msg
            # Try to parse "mat1 and mat2 shapes cannot be multiplied (1xH and N x M)"
            # to recover N (expected in_features).
            # Common forms:
            #  - "mat1 and mat2 shapes cannot be multiplied (1x128 and 256x64)"
            #  - "einsum(): subscript ... (got 128) ... (expected 256)" (we ignore)
            import re
            m = re.search(r"cannot be multiplied\s*\(\s*\d+x(\d+)\s*and\s*(\d+)x\d+\)", msg)
            if m:
                got_H = int(m.group(1))
                need_N = int(m.group(2))
                try:
                    out = try_with_H(need_N)
                    print(f"[test_tcp] inferred state dim via matmul error: H={need_N} (was {got_H})")
                    return out
                except Exception as e2:
                    # if that didn't work, keep looping
                    first_error = str(e2)

    # If we’re here, none of the quick/inferred dims worked. Give one more helpful error.
    raise RuntimeError(
        "Could not execute forward with any tried state dimension. "
        "Model expects a Tensor `state` of shape [1,H] matching a Linear's in_features.\n"
        f"First failure was: {first_error or 'N/A'}"
    )


def main() -> int:
    if not os.path.isfile(CONFIG_PATH):
        print(f"[test_tcp] config not found: {CONFIG_PATH}"); return 2
    if not os.path.isfile(IMAGE_PATH):
        print(f"[test_tcp] image not found: {IMAGE_PATH}"); return 2

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    if "tcp" not in cfg:
        print("[test_tcp] 'tcp' section missing in config.yaml"); return 2
    tcp = cfg["tcp"]

    tp_xy = TARGET_POINT_XY
    if tp_xy is None:
        # use aim_dist from config.tcp (fallback 4.0) and center laterally
        tp_xy = (float(tcp.get("aim_dist", 4.0)), 0.0)

    model, device, cmd_dim = build_model_from_tcp(tcp)
    in_w, in_h = int(tcp.get("input_w", 256)), int(tcp.get("input_h", 256))
    img = load_image_tensor(IMAGE_PATH, (in_w, in_h))

    out = forward_one(model, img, CMD_STR, SPEED_MPS, tp_xy, device, cmd_dim)


    # Print raw output
    print("=== RAW OUTPUT ===")
    if torch.is_tensor(out):
        print(repr(out.detach().cpu()))
    else:
        try:
            print(repr(out))
        except Exception:
            print(type(out))

    # Quick summary if it looks like controls or waypoints
    try:
        if torch.is_tensor(out):
            t = out.detach().cpu()
            if t.ndim == 3 and t.shape[-1] == 2:
                wp = t[0,0]
                x_wp, y_wp = float(wp[0]), float(wp[1])
                steer = float(np.clip(np.arctan2(y_wp, max(1e-3, x_wp)) / 0.4, -1.0, 1.0))
                target_v = 6.0; v_err = target_v - SPEED_MPS
                throttle = float(np.clip(0.2 * v_err, 0.0, 1.0))
                brake = 0.0 if v_err >= 0 else float(np.clip(-0.3 * v_err, 0.0, 1.0))
                print("\n(interpreted as controls from waypoints) "
                      f"steer={steer:.3f} throttle={throttle:.3f} brake={brake:.3f}")
            elif t.ndim >= 2 and t.shape[-1] >= 3:
                y = t.reshape(-1, t.shape[-1])[0].float().numpy()
                print("\n(first 3 as controls) "
                      f"steer={float(y[0]):.3f} throttle={float(y[1]):.3f} brake={float(y[2]):.3f}")
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
