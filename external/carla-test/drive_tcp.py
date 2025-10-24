#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drive_tcp.py — Drive CARLA using a TCP model (Lightning .ckpt)

Features
--------
- Training-accurate target_point transform (matches TCP/data.py).
- Two control modes:
    • "pid"     — robust: predicted waypoints + model's built-in PID controller
    • "policy"  — direct actions from the policy head (mu/sigma)
- Optional one-time PID axes calibration (configurable).
- Clean logs and explicit CARLA control flags.

Config
------
Use the same YAML structure you already have (see example in your repo), plus:

tcp:
  ckpt: ~/Downloads/best_model.ckpt
  device: cpu                 # "cpu" or "cuda"
  control: pid                # "pid" | "policy"
  speed_div: 12.0             # speed normalization divisor used in training
  lookahead_m: 12.0           # distance ahead for target_point
  stabilize_s: 2.0            # initial seconds forcing "lanefollow" command
  pid_axes: auto              # "auto" | "xy" | "negx" | "negy" | "swap" | "swap_negx" | "swap_negy"

Run
---
python drive_tcp.py --config config.yaml
"""

from __future__ import annotations
import argparse
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as T
import carla  # type: ignore

# --- Local helpers (repo) ---
from sim_utils import (
    set_sync, attach_rgb, set_weather, update_spectator, destroy_actors, town_basename,
)
from agents_helpers import build_grp, pick_routes, next_high_level_command

# --- TCP model ---
from TCP.model import TCP
from TCP.config import GlobalConfig


# ---------------- Constants ----------------
HOST, RPC_PORT, TM_PORT = "localhost", 2000, 8000

# Image size that yields cnn_feature of (H,W)=(8,29) for ResNet-34 backbone
IMG_H, IMG_W = 256, 928

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CMD_NAMES = ["left","right","straight","lanefollow","changelaneleft","changelaneright"]

PRINT_EVERY = 10  # tick logging cadence


# ---------------- Small helpers ----------------
def _to_bool(x) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","on","y","t"}
    return bool(x)

def _validate_config(cfg: Dict) -> None:
    required_top = ["town","fps","seed","duration_s","ego_blueprint","traffic_vehicles","camera","view"]
    for k in required_top:
        if k not in cfg: raise ValueError(f"Missing config key: {k}")
    cam_req = ["x","y","z","roll","pitch","yaw","image_size_x","image_size_y","fov"]
    for k in cam_req:
        if k not in cfg["camera"]: raise ValueError(f"Missing camera config key: camera.{k}")
    if "preset" not in (cfg.get("weather") or {}): raise ValueError("Missing weather.preset")
    if int(cfg["fps"]) <= 0: raise ValueError("fps must be > 0")
    if float(cfg["duration_s"]) <= 0: raise ValueError("duration_s must be > 0")

def load_ckpt_into_tcp(model: nn.Module, ckpt_path: str, strict: bool = False):
    """Load a Lightning .ckpt where weights live under 'model.' prefix (TCP_planner)."""
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        raise RuntimeError(f"No 'state_dict' in checkpoint: {list(ckpt.keys())}")
    new_sd = {}
    for k, v in ckpt["state_dict"].items():
        new_sd[k[len("model."):]] = v if k.startswith("model.") else v
    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if missing or unexpected:
        print(f"[drive_tcp] load_state_dict: missing={missing}, unexpected={unexpected}")

def _cmd_to_index(cmd) -> int:
    """
    Map RoadOption (enum/name) or string/int to training indices 0..5 (after `command -= 1`).
    """
    name = str(cmd).lower()
    map_enum = {
        "roadoption.left": "left",
        "roadoption.right": "right",
        "roadoption.straight": "straight",
        "roadoption.lanefollow": "lanefollow",
        "roadoption.changelaneleft": "changelaneleft",
        "roadoption.changelaneright": "changelaneright",
    }
    name = map_enum.get(name, name)
    if name in CMD_NAMES:
        return CMD_NAMES.index(name)
    try:
        val = int(cmd)
        if 1 <= val <= 6:   # training did command -= 1
            return val - 1
        if 0 <= val <= 5:
            return val
    except Exception:
        pass
    return 3  # default to lanefollow

def _one_hot(idx: int) -> torch.Tensor:
    idx = int(np.clip(idx, 0, 5))
    v = torch.zeros(6, dtype=torch.float32)
    v[idx] = 1.0
    return v

def _carla_img_to_rgb_tensor(img: "carla.Image", size_hw: Tuple[int,int]) -> torch.Tensor:
    """Convert CARLA BGRA image → normalized RGB tensor (1,3,H,W)."""
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    rgb = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    pil = Image.fromarray(rgb)
    tfm = T.Compose([T.Resize(size_hw), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return tfm(pil).unsqueeze(0)

# ---------- Training-accurate target_point transform ----------
def tp_in_ego_training_convention(ego_tf: carla.Transform, target_loc: carla.Location) -> Tuple[float,float]:
    """
    Match TCP/data.py:
      R = R(π/2 + yaw)
      v = [y_target - ego_y,  x_target - ego_x]
      tp = R^T @ v
    """
    ex, ey = ego_tf.location.x, ego_tf.location.y
    dy = target_loc.y - ey
    dx = target_loc.x - ex
    yaw = math.radians(ego_tf.rotation.yaw)
    c = math.cos(math.pi/2 + yaw); s = math.sin(math.pi/2 + yaw)
    tp_x =  c * dy + s * dx
    tp_y = -s * dy + c * dx
    return float(tp_x), float(tp_y)

# ---------- PID axes helpers ----------
def _apply_xy_transform(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Deterministic axis transform for PID-only (for forks that flip axes)."""
    y = x.clone()
    if kind == "xy": return y
    if kind == "negx": y[...,0] = -y[...,0]; return y
    if kind == "negy": y[...,1] = -y[...,1]; return y
    if kind == "swap": y[...,0], y[...,1] = y[...,1].clone(), y[...,0].clone(); return y
    if kind == "swap_negx": y[...,0], y[...,1] = y[...,1].clone(), -y[...,0].clone(); return y
    if kind == "swap_negy": y[...,0], y[...,1] = -y[...,1].clone(), y[...,0].clone(); return y
    return y

def _auto_calibrate_pid_axes(model: TCP, img_t: torch.Tensor,
                             speed_mps: float, cmd_idx: int,
                             tp_raw: torch.Tensor) -> Tuple[str, str]:
    """
    Probe a single forward pass and pick (wp_kind, tp_kind) that yields the smallest |steer|.
    This is cheap and runs once. If you prefer deterministic behavior, set pid_axes != "auto".
    """
    with torch.no_grad():
        speed_norm = torch.tensor([[min(max(speed_mps/12.0, 0.0), 1.0)]], dtype=torch.float32, device=img_t.device)
        cmd_oh = _one_hot(cmd_idx).view(1, -1).to(img_t.device)
        state = torch.cat([speed_norm, tp_raw, cmd_oh], dim=1)
        pred = model(img_t, state, tp_raw)
        pred_wp = pred["pred_wp"]
        speed_raw = torch.tensor([[speed_mps]], dtype=torch.float32, device=img_t.device)

        kinds = ["xy","negx","negy","swap","swap_negx","swap_negy"]
        best = ("xy", "xy", 1e9)
        for wk in kinds:
            wpt = _apply_xy_transform(pred_wp, wk)
            for tk in kinds:
                tpt = _apply_xy_transform(tp_raw, tk)
                steer, _, _, _ = model.control_pid(wpt, speed_raw, tpt)
                s = abs(float(steer))
                if s < best[2]:
                    best = (wk, tk, s)
        return best[0], best[1]


# ---------------- Main ----------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Drive CARLA with TCP")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.config):
        print(f"[drive_tcp] Config not found: {args.config}", file=sys.stderr)
        return 2

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    try:
        _validate_config(cfg)
    except Exception as e:
        print(f"[drive_tcp] Bad config: {e}", file=sys.stderr); return 2

    # TCP/runtime params
    tcp_cfg = (cfg.get("tcp") or {})
    CKPT        = os.path.expanduser(tcp_cfg.get("ckpt", "~/Downloads/best_model.ckpt"))
    DEVICE      = str(tcp_cfg.get("device", "cpu")).lower()
    CONTROL     = str(tcp_cfg.get("control", "pid")).lower()            # "pid" | "policy"
    SPEED_DIV   = float(tcp_cfg.get("speed_div", 12.0))
    LOOKAHEAD_M = float(tcp_cfg.get("lookahead_m", 12.0))
    STABILIZE_S = float(tcp_cfg.get("stabilize_s", 2.0))
    PID_AXES    = str(tcp_cfg.get("pid_axes", "auto")).lower()          # "auto" | fixed kind

    # Connect to CARLA
    client = carla.Client(HOST, RPC_PORT)
    client.set_timeout(20.0)
    world: "carla.World" = client.get_world()

    # Load target town if different
    current = town_basename(world.get_map().name)
    target  = str(cfg["town"])
    if current != target:
        time.sleep(1.0)
        client.load_world(target)
        world = client.get_world()

    tm: "carla.TrafficManager" = client.get_trafficmanager(TM_PORT)

    # Seeds + mode
    SEED = int(cfg["seed"])
    random.seed(SEED); np.random.seed(SEED)
    tm.set_random_device_seed(SEED)
    world.set_pedestrians_seed(SEED + 1)

    FPS = int(cfg["fps"])
    SYNC = str((cfg.get("runtime") or {}).get("mode", "sync")).lower() == "sync"
    set_sync(world, tm, FPS, SYNC)

    # Warm-up & weather
    if SYNC:
        for _ in range(30): world.tick()
    else:
        for _ in range(30): world.wait_for_tick(seconds=1.0)
    set_weather(world, cfg["weather"]["preset"])

    # Spawn ego
    sps = world.get_map().get_spawn_points()
    sps.sort(key=lambda t: (t.location.x, t.location.y, t.location.z))
    ego_bp = world.get_blueprint_library().find(cfg["ego_blueprint"])
    ego = world.try_spawn_actor(ego_bp, sps[0])
    if not ego:
        raise RuntimeError("Failed to spawn ego at first spawn point.")
    actors: List["carla.Actor"] = [ego]

    # Optional background traffic
    veh_bps = list(world.get_blueprint_library().filter("vehicle.*"))
    i = 1
    while i < len(sps) and len(actors) - 1 < int(cfg["traffic_vehicles"]):
        bp = veh_bps[(len(actors)-1) % len(veh_bps)]
        v = world.try_spawn_actor(bp, sps[i]); i += 1
        if v:
            v.set_autopilot(True, tm.get_port())
            actors.append(v)

    # Attach RGB camera
    cam, q = attach_rgb(world, ego, cfg["camera"])
    actors.append(cam)

    # Spectator & first frame
    spectator = world.get_spectator()
    for _ in range(120):
        if SYNC: world.tick()
        else:    world.wait_for_tick(seconds=1.0)
        update_spectator(spectator, ego, cam, cfg.get("view","sensor"))
        if len(q) > 0: break
    else:
        raise RuntimeError("Camera never produced a frame in warm-up")

    # TCP model
    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
    tcp_model = TCP(GlobalConfig()).to(device).eval()
    load_ckpt_into_tcp(tcp_model, CKPT, strict=False)

    # Perception stride sanity (optional informational print)
    with torch.no_grad():
        _, feat = tcp_model.perception(_carla_img_to_rgb_tensor(q[-1], (IMG_H, IMG_W)).to(device))
    print(f"[drive_tcp] perception feature HW = {tuple(feat.shape[-2:])} (expect (8, 29))")

    # Routes for high-level commands
    grp = build_grp(world, sampling_resolution=2.0)
    rng = random.Random(SEED)
    routes = pick_routes(
        world, grp,
        int(cfg.get("routes", {}).get("num_routes", 5)),
        float(cfg.get("routes", {}).get("min_route_m", 500.0)),
        float(cfg.get("routes", {}).get("max_route_m", 1500.0)),
        rng,
    )
    route_idx, step_in_route = 0, 0

    # One-time PID axes mode (auto or fixed)
    wp_kind, tp_kind = "xy", "xy"
    if CONTROL == "pid":
        # Build one small probe (lanefollow)
        tr0 = ego.get_transform()
        wmap = world.get_map()
        wp0 = wmap.get_waypoint(tr0.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        nxt0 = wp0.next(max(LOOKAHEAD_M, 1e-3))
        tgt0 = (nxt0[0].transform.location if nxt0 else wp0.transform.location)
        tp_x0, tp_y0 = tp_in_ego_training_convention(tr0, tgt0)
        tp_raw0 = torch.tensor([[tp_x0, tp_y0]], dtype=torch.float32, device=device)

        spd0 = ego.get_velocity(); spd0_mps = float((spd0.x**2 + spd0.y**2 + spd0.z**2)**0.5)
        img_t0 = _carla_img_to_rgb_tensor(q[-1], (IMG_H, IMG_W)).to(device)

        if PID_AXES == "auto":
            wp_kind, tp_kind = _auto_calibrate_pid_axes(tcp_model, img_t0, spd0_mps, _cmd_to_index("lanefollow"), tp_raw0)
        else:
            wp_kind, tp_kind = PID_AXES, PID_AXES
        print(f"[drive_tcp] PID axes: wp='{wp_kind}', tp='{tp_kind}'")

    # Loop control
    tick_idx = 0
    snap0 = world.get_snapshot()
    sim_start = snap0.timestamp.elapsed_seconds
    end_time = sim_start + float(cfg["duration_s"])

    FAST = _to_bool((cfg.get("runtime") or {}).get("fast_mode", False))
    if SYNC and not FAST:
        dt_wall = 1.0 / FPS
        next_wall = time.perf_counter()

    try:
        while True:
            snap = world.get_snapshot()
            if snap and snap.timestamp.elapsed_seconds >= end_time:
                break

            if not q:
                if SYNC: world.tick()
                else:    world.wait_for_tick(seconds=1.0)
                continue
            img = q[-1]

            # Measurements
            tr = ego.get_transform()
            v = ego.get_velocity()
            speed_mps = float((v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5)

            # High-level command (stabilize as lanefollow briefly)
            sim_t = snap.timestamp.elapsed_seconds
            route = routes[route_idx]
            raw_cmd = "lanefollow" if (sim_t - sim_start) < STABILIZE_S else next_high_level_command(route, step_in_route)
            cmd_idx = _cmd_to_index(raw_cmd)
            cmd_oh  = _one_hot(cmd_idx).view(1, -1).to(device)

            # Target point in training convention
            wmap = world.get_map()
            wp = wmap.get_waypoint(tr.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            nxt = wp.next(max(LOOKAHEAD_M, 1e-3))
            tgt_loc = (nxt[0].transform.location if nxt else wp.transform.location)
            tp_x, tp_y = tp_in_ego_training_convention(tr, tgt_loc)
            tp_raw = torch.tensor([[tp_x, tp_y]], dtype=torch.float32, device=device)

            # Build inputs
            x_img = _carla_img_to_rgb_tensor(img, (IMG_H, IMG_W)).to(device)
            speed_norm = torch.tensor([[min(max(speed_mps/float(SPEED_DIV), 0.0), 1.0)]],
                                      dtype=torch.float32, device=device)
            state = torch.cat([speed_norm, tp_raw, cmd_oh], dim=1)

            # Forward + control
            with torch.no_grad():
                pred = tcp_model(x_img, state, tp_raw)

                if CONTROL == "pid":
                    # Use model waypoints + builtin PID
                    wp_pid = _apply_xy_transform(pred["pred_wp"], wp_kind)
                    tp_pid = _apply_xy_transform(tp_raw,        tp_kind)
                    speed_raw = torch.tensor([[speed_mps]], dtype=torch.float32, device=device)
                    steer_t, thr_t, brk_t, _ = tcp_model.control_pid(wp_pid, speed_raw, tp_pid)
                    steer   = float(np.clip(steer_t, -1.0, 1.0))
                    throttle= float(np.clip(thr_t, 0.0, 1.0))
                    brake   = float(np.clip(brk_t, 0.0, 1.0))
                else:
                    # Direct policy actions (mu/sigma -> action)
                    mu, sigma = pred["mu_branches"], pred["sigma_branches"]
                    thr_t, steer_t, brk_t = tcp_model.get_action(mu[0], sigma[0])
                    steer   = float(np.clip(steer_t.item(), -1.0, 1.0))
                    throttle= float(np.clip(thr_t.item(), 0.0, 1.0))
                    brake   = float(np.clip(brk_t.item(), 0.0, 1.0))

            

            # Logging
            if (tick_idx % PRINT_EVERY) == 0:
                print(
                    f"[tcp] mode={CONTROL:6s} | t={sim_t-sim_start:5.2f}s | spd={speed_mps:5.2f} "
                    f"(norm={speed_mps/SPEED_DIV:.2f}) | cmd={CMD_NAMES[cmd_idx]} | "
                    f"tp=({tp_x:.2f},{tp_y:.2f}) | applied: thr={throttle:.3f} steer={steer:.3f} brk={brake:.3f}"
                )

            # Apply control (explicit flags)
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1,
            )
            ego.apply_control(control)

            # Advance world
            if SYNC:
                if not FAST:
                    next_wall += dt_wall
                    sleep = next_wall - time.perf_counter()
                    if sleep > 0: time.sleep(sleep)
                    else: next_wall = time.perf_counter()
                world.tick()
            else:
                world.wait_for_tick(seconds=1.0)

            tick_idx += 1
            update_spectator(spectator, ego, cam, cfg.get("view","sensor"))

            # Route bookkeeping
            tgt_end = route[-1][0].transform.location
            if tr.location.distance(tgt_end) < 5.0:
                route_idx = (route_idx + 1) % len(routes)
                step_in_route = 0
            else:
                step_in_route += 1

        print("[drive_tcp] Finished driving.")
        return 0

    except KeyboardInterrupt:
        print("\n[drive_tcp] Stopped by user.")
        return 0
    except Exception as e:
        print(f"[drive_tcp] Error: {e}", file=sys.stderr)
        return 1
    finally:
        # Stop sensors, destroy actors, leave server async
        try:
            for a in actors:
                if a and a.type_id.startswith("sensor."):
                    try: a.stop()
                    except Exception: pass
        except Exception:
            pass
        destroy_actors(reversed(actors))
        try: set_sync(world, tm, FPS, False)
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
