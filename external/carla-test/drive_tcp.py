#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drive_tcp.py — Drive CARLA using a TCP model (Lightning ckpt).

What this script ensures
------------------------
- Ego spawn is lane-centered & yaw-aligned (no initial drift).
- Target point (TP) is on the lane center, LOOKAHEAD_M meters ahead.
- Correct, single world→ego transform: +x forward, +y right.
- Optional debug overlays: forward arrow & TP dot (in UE window).
- Control modes:
    * policy : TCP action head
    * pid    : TCP PID controller over predicted waypoints
- Runtime: sync/async + fast mode (sync only)

Config keys used (same style as collector.py)
---------------------------------------------
town, fps, seed, duration_s, ego_blueprint, traffic_vehicles,
camera{...}, weather.preset, view ("sensor"|"chase"),
runtime.mode ("sync"|"async"), runtime.fast_mode (bool),
tcp.ckpt, tcp.device, tcp.control ("policy"|"pid"),
tcp.speed_div, tcp.lookahead_m,
debug.enabled (bool)
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

# --- Local helpers (from your repo) ---
from sim_utils import (
    set_sync, attach_rgb, set_weather, update_spectator, destroy_actors, town_basename,
)
from agents_helpers import build_grp, pick_routes, next_high_level_command

# --- TCP model ---
from TCP.model import TCP
from TCP.config import GlobalConfig

# ---------------- Constants ----------------
HOST = "localhost"
RPC_PORT = 2000
TM_PORT = 8000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Input size measured for your backbone stride (per your earlier logs)
IMG_H, IMG_W = 256, 928

CMD_NAMES = ["left","right","straight","lanefollow","changelaneleft","changelaneright"]
PRINT_EVERY = 10   # print every N ticks

# ---------------- Small helpers ----------------

def _to_bool(x) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","on","y","t"}
    return bool(x)

def _validate_config(cfg: Dict) -> None:
    required_top = ["town","fps","seed","duration_s","ego_blueprint","traffic_vehicles","camera","view"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"Missing config key: {k}")
    cam_req = ["x","y","z","roll","pitch","yaw","image_size_x","image_size_y","fov"]
    for k in cam_req:
        if k not in cfg["camera"]:
            raise ValueError(f"Missing camera config key: camera.{k}")
    if "preset" not in (cfg.get("weather") or {}):
        raise ValueError("Missing weather.preset")
    if int(cfg["fps"]) <= 0: raise ValueError("fps must be > 0")
    if float(cfg["duration_s"]) <= 0: raise ValueError("duration_s must be > 0")

def load_ckpt_into_tcp(model: nn.Module, ckpt_path: str, strict: bool = False):
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        raise RuntimeError(f"No 'state_dict' in checkpoint: {list(ckpt.keys())}")
    sd = ckpt["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        new_sd[k[len("model."):] if k.startswith("model.") else k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if missing or unexpected:
        print(f"[drive_tcp] load_state_dict: missing={missing}, unexpected={unexpected}")

def _one_hot_command(name_or_idx) -> torch.Tensor:
    if isinstance(name_or_idx, int):
        idx = name_or_idx
    else:
        name = str(name_or_idx).lower()
        idx = CMD_NAMES.index(name) if name in CMD_NAMES else 3  # lanefollow fallback
    idx = int(np.clip(idx, 0, 5))
    oh = torch.zeros(6, dtype=torch.float32)
    oh[idx] = 1.0
    return oh

def _carla_img_to_rgb_tensor(img: "carla.Image", size_hw: Tuple[int,int]) -> torch.Tensor:
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    rgb = arr[:, :, :3][:, :, ::-1]  # BGRA->RGB
    pil = Image.fromarray(rgb)
    tfm = T.Compose([
        T.Resize(size_hw),  # (H,W)
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(pil).unsqueeze(0)

# ---------- Geometry (clean, single convention) ----------

def world_to_ego_xy(world_loc: carla.Location, ego_tf: carla.Transform) -> Tuple[float,float]:
    """
    World → ego 2D (meters). Ego frame: +x forward, +y right.
    """
    dx = world_loc.x - ego_tf.location.x
    dy = world_loc.y - ego_tf.location.y
    yaw = math.radians(ego_tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    x_ego =  c * dx + s * dy
    y_ego = -s * dx + c * dy
    return float(x_ego), float(y_ego)

def ego_forward_tip(ego_tf: carla.Transform, dist: float) -> carla.Location:
    yaw = math.radians(ego_tf.rotation.yaw)
    return carla.Location(
        x=ego_tf.location.x + dist * math.cos(yaw),
        y=ego_tf.location.y + dist * math.sin(yaw),
        z=ego_tf.location.z,
    )

def ego_lane_ahead(world_map: "carla.Map", ego_tf: carla.Transform, lookahead_m: float) -> carla.Location:
    wp = world_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    nxt = wp.next(max(1e-3, float(lookahead_m)))
    return (nxt[0].transform.location if nxt else wp.transform.location)

def clip_tp(tp_x: float, tp_y: float, clip_m: float = 30.0) -> Tuple[float,float]:
    return float(np.clip(tp_x, -clip_m, clip_m)), float(np.clip(tp_y, -clip_m, clip_m))

# ---------- Spawn helpers ----------

def choose_lane_center_spawn(world: "carla.World") -> carla.Transform:
    """
    Prefer a spawn whose nearest waypoint is not in a junction and remains
    lanefollow shortly ahead. Fall back to first spawn if none found.
    """
    wmap = world.get_map()
    sps = sorted(wmap.get_spawn_points(), key=lambda t: (t.location.x, t.location.y, t.location.z))
    for sp in sps:
        wp = wmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp and not wp.is_junction:
            nxt = wp.next(30.0)
            if nxt and not nxt[0].is_junction:
                return wp.transform
    # fallback
    return wmap.get_waypoint(sps[0].location, project_to_road=True, lane_type=carla.LaneType.Driving).transform

# ---------- PID convention calibration (kept minimal/safe) ----------

def _apply_xy_transform(x: torch.Tensor, kind: str) -> torch.Tensor:
    """
    Apply convention transform to a (..,2) tensor; used ONLY for PID alignment
    with the TCP model's expected (learned) waypoint axes.
    """
    y = x.clone()
    if kind == "xy": return y
    if kind == "negx": y[...,0] = -y[...,0]; return y
    if kind == "negy": y[...,1] = -y[...,1]; return y
    if kind == "swap": y[...,0], y[...,1] = y[...,1].clone(), y[...,0].clone(); return y
    if kind == "swap_negx": y[...,0], y[...,1] = y[...,1].clone(), -y[...,0].clone(); return y
    if kind == "swap_negy": y[...,0], y[...,1] = -y[...,1].clone(), y[...,0].clone(); return y
    raise ValueError(kind)

def calibrate_pid_axes(model: TCP, speed_raw: torch.Tensor, pred_wp: torch.Tensor, tp_raw: torch.Tensor) -> Tuple[str, str, float]:
    """
    Try a small grid of transforms; pick the pair with minimum |steer|.
    This DOES NOT change the main world→ego transform; it's only to match
    the model's trained waypoint axes if they differ.
    """
    kinds = ["xy","negx","negy","swap","swap_negx","swap_negy"]
    best = None
    with torch.no_grad():
        for wk in kinds:
            wpt = _apply_xy_transform(pred_wp, wk)
            for tk in kinds:
                tpt = _apply_xy_transform(tp_raw, tk)
                steer, throttle, brake, _ = model.control_pid(wpt, speed_raw, tpt)
                score = abs(float(steer))
                if (best is None) or (score < best[2]):
                    best = (wk, tk, score)
    return best  # type: ignore

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
        print(f"[drive_tcp] Bad config: {e}", file=sys.stderr)
        return 2

    # TCP/runtime params
    tcp_cfg = (cfg.get("tcp") or {})
    CKPT = os.path.expanduser(tcp_cfg.get("ckpt", "~/Downloads/best_model.ckpt"))
    DEVICE = str(tcp_cfg.get("device", "cpu")).lower()
    CONTROL_MODE = str(tcp_cfg.get("control", "policy")).lower()  # "policy" | "pid"
    SPEED_DIV = float(tcp_cfg.get("speed_div", 12.0))
    LOOKAHEAD_M = float(tcp_cfg.get("lookahead_m", 12.0))
    STABILIZE_S = float(tcp_cfg.get("stabilize_s", 2.0))  # force LANEFOLLOW early

    runtime_cfg = (cfg.get("runtime") or {})
    SYNC = str(runtime_cfg.get("mode", "sync")).lower() == "sync"
    FAST = _to_bool(runtime_cfg.get("fast_mode", False))
    FPS = int(cfg["fps"])

    dbg_cfg = (cfg.get("debug") or {})
    DEBUG_DRAW = _to_bool(dbg_cfg.get("enabled", False))

    # Connect
    client = carla.Client(HOST, RPC_PORT)
    client.set_timeout(20.0)
    world: "carla.World" = client.get_world()

    # Map (load if different)
    current = town_basename(world.get_map().name)
    target = str(cfg["town"])
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
    set_sync(world, tm, FPS, SYNC)

    # Warm-up a bit (gets the sim stable)
    if SYNC:
        for _ in range(30): world.tick()
    else:
        for _ in range(30): world.wait_for_tick(seconds=1.0)

    # Weather
    set_weather(world, cfg["weather"]["preset"])

    # Spawn ego (lane-centered) + optional traffic
    ego_bp = world.get_blueprint_library().find(cfg["ego_blueprint"])
    spawn_tf = choose_lane_center_spawn(world)
    ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if not ego:
        raise RuntimeError("Failed to spawn ego at chosen lane-centered transform.")

    actors: List["carla.Actor"] = [ego]

    # Traffic
    sps = world.get_map().get_spawn_points()
    sps.sort(key=lambda t: (t.location.x, t.location.y, t.location.z))
    veh_bps = list(world.get_blueprint_library().filter("vehicle.*"))
    i = 0
    while i < len(sps) and len(actors) - 1 < int(cfg["traffic_vehicles"]):
        bp = veh_bps[(len(actors)-1) % len(veh_bps)]
        v = world.try_spawn_actor(bp, sps[i]); i += 1
        if v:
            v.set_autopilot(True, tm.get_port())
            actors.append(v)

    # Camera
    cam, q = attach_rgb(world, ego, cfg["camera"])
    actors.append(cam)

    # Spectator & ensure first frame exists
    spectator = world.get_spectator()
    for _ in range(120):
        if SYNC: world.tick()
        else: world.wait_for_tick(seconds=1.0)
        update_spectator(spectator, ego, cam, cfg.get("view","sensor"))
        if len(q) > 0:
            break
    else:
        raise RuntimeError("Camera never produced a frame in warm-up")

    # TCP model
    device = torch.device(DEVICE if (DEVICE=="cpu" or torch.cuda.is_available()) else "cpu")
    tcp_model = TCP(GlobalConfig()).to(device).eval()
    load_ckpt_into_tcp(tcp_model, CKPT, strict=False)

    # (One-shot) perception stride sanity check
    x_probe = _carla_img_to_rgb_tensor(q[-1], (IMG_H, IMG_W)).to(device)
    with torch.no_grad():
        _, feat = tcp_model.perception(x_probe)
    print(f"[drive_tcp] perception feature HW = {tuple(feat.shape[-2:])} (expect (8, 29))")

    # Routes
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

    # One-time PID axes calibration (kept, but you’ll see a warning if not identity)
    # Build a clean TP from lane center ahead, using our world→ego transform.
    ego_tf0 = ego.get_transform()
    world_map = world.get_map()
    tgt0 = ego_lane_ahead(world_map, ego_tf0, LOOKAHEAD_M)
    tp_x0, tp_y0 = world_to_ego_xy(tgt0, ego_tf0)
    tp_x0, tp_y0 = clip_tp(tp_x0, tp_y0, 30.0)
    tp_raw0 = torch.tensor([[tp_x0, tp_y0]], dtype=torch.float32, device=device)

    vel0 = ego.get_velocity()
    speed0 = float((vel0.x**2 + vel0.y**2 + vel0.z**2) ** 0.5)
    speed_raw0 = torch.tensor([[speed0]], dtype=torch.float32, device=device)

    # Need a pred_wp for calibration:
    with torch.no_grad():
        speed_norm0 = torch.tensor([[min(max(speed0/12.0, 0.0), 1.0)]], dtype=torch.float32, device=device)
        cmd_oh0 = _one_hot_command("lanefollow").view(1, -1).to(device)
        state0 = torch.cat([speed_norm0, tp_raw0, cmd_oh0], dim=1)
        pred0 = tcp_model(x_probe, state0, tp_raw0)
        pred_wp0 = pred0["pred_wp"]  # (1,T,2)

    WP_KIND, TP_KIND, steer0 = calibrate_pid_axes(tcp_model, speed_raw0, pred_wp0, tp_raw0)
    note = "" if (WP_KIND,TP_KIND)==("xy","xy") else "  (NOTE: model uses non-identity axes; applying safe remap)"
    print(f"[drive_tcp] PID calibration: wp='{WP_KIND}', tp='{TP_KIND}', |steer|≈{steer0:.3f}{note}")

    # Main loop
    tick_idx = 0
    snap0 = world.get_snapshot()
    start_sim_t = snap0.timestamp.elapsed_seconds
    end_time = start_sim_t + float(cfg["duration_s"])

    # Pacing (sync & not fast)
    if SYNC and not FAST:
        dt_wall = 1.0 / FPS
        next_wall = time.perf_counter()

    try:
        while True:
            snap = world.get_snapshot()
            if snap and snap.timestamp.elapsed_seconds >= end_time:
                break

            # Get latest frame; if none, tick & continue
            if not q:
                if SYNC: world.tick()
                else: world.wait_for_tick(seconds=1.0)
                continue
            img = q[-1]

            # Measurements
            tr = ego.get_transform()
            vel = ego.get_velocity()
            speed_mps = float((vel.x*vel.x + vel.y*vel.y + vel.z*vel.z) ** 0.5)

            # High-level command; stabilize early seconds as lanefollow
            sim_t = snap.timestamp.elapsed_seconds
            route = routes[route_idx]
            cmd = "lanefollow" if (sim_t - start_sim_t) < STABILIZE_S else next_high_level_command(route, step_in_route)

            # Target point: lane-ahead in world → ego frame (clean)
            tgt_loc = ego_lane_ahead(world.get_map(), tr, LOOKAHEAD_M)
            tp_x_raw, tp_y_raw = world_to_ego_xy(tgt_loc, tr)
            tp_x_raw, tp_y_raw = clip_tp(tp_x_raw, tp_y_raw, clip_m=30.0)

            # Image tensor
            x_img = _carla_img_to_rgb_tensor(img, (IMG_H, IMG_W)).to(device)

            # Build state (speed normalized to [0,1])
            speed_norm = torch.tensor([[min(max(speed_mps/float(SPEED_DIV), 0.0), 1.0)]],
                                      dtype=torch.float32, device=device)
            cmd_oh = _one_hot_command(cmd).view(1, -1).to(device)
            tp_raw = torch.tensor([[tp_x_raw, tp_y_raw]], dtype=torch.float32, device=device)
            state = torch.cat([speed_norm, tp_raw, cmd_oh], dim=1)

            # Forward
            with torch.no_grad():
                pred = tcp_model(x_img, state, tp_raw)

                if CONTROL_MODE == "pid":
                    # Remap to model’s learned axes (from calibration)
                    wp_pid = _apply_xy_transform(pred["pred_wp"], WP_KIND)
                    tp_pid = _apply_xy_transform(tp_raw, TP_KIND)
                    speed_raw = torch.tensor([[speed_mps]], dtype=torch.float32, device=device)
                    steer_t, thr_t, brk_t, _ = tcp_model.control_pid(wp_pid, speed_raw, tp_pid)
                    steer   = float(steer_t)
                    throttle= float(thr_t)
                    brake   = float(brk_t)
                else:
                    mu, sigma = pred["mu_branches"], pred["sigma_branches"]
                    thr_t, steer_t, brk_t = tcp_model.get_action(mu[0], sigma[0])
                    steer   = float(steer_t.item())
                    throttle= float(thr_t.item())
                    brake   = float(brk_t.item())

            # Print
            if (tick_idx % PRINT_EVERY) == 0:
                cmd_idx = int(torch.argmax(cmd_oh, dim=1).item())
                if CONTROL_MODE == "policy":
                    mu_np = mu[0].detach().cpu().numpy(); sg_np = sigma[0].detach().cpu().numpy()
                    print(
                        f"[tcp] mode=policy | spd={speed_mps:5.2f} (norm={speed_mps/float(SPEED_DIV):.2f}) | "
                        f"cmd={CMD_NAMES[cmd_idx]} | tp=({tp_x_raw:.2f},{tp_y_raw:.2f}) | "
                        f"mu={mu_np.round(3)} sigma={sg_np.round(3)} | "
                        f"applied: thr={throttle:.3f} steer={steer:.3f} brk={brake:.3f}"
                    )
                else:
                    print(
                        f"[tcp] mode=pid    | spd={speed_mps:5.2f} | cmd={CMD_NAMES[cmd_idx]} | "
                        f"tp=({tp_x_raw:.2f},{tp_y_raw:.2f}) [axes {TP_KIND}] | "
                        f"WP axes={WP_KIND} | applied: thr={throttle:.3f} steer={steer:.3f} brk={brake:.3f}"
                    )

            # Apply control
            control = carla.VehicleControl(
                throttle=float(np.clip(throttle, 0.0, 1.0)),
                steer=float(np.clip(steer, -1.0, 1.0)),
                brake=float(np.clip(brake, 0.0, 1.0)),
            )
            ego.apply_control(control)

            # Debug overlays
            if DEBUG_DRAW:
                fwd_tip = ego_forward_tip(tr, 8.0)
                world.debug.draw_arrow(tr.location, fwd_tip, thickness=0.05,
                                       arrow_size=0.2, color=carla.Color(0,255,0), life_time=0.1)
                world.debug.draw_point(tgt_loc, 0.08, carla.Color(0,0,255), 0.2, False)

            # Advance world
            if SYNC:
                if not FAST:
                    # pace to FPS
                    next_wall += 1.0 / FPS
                    sleep = next_wall - time.perf_counter()
                    if sleep > 0: time.sleep(sleep)
                    else: next_wall = time.perf_counter()
                world.tick()
            else:
                world.wait_for_tick(seconds=1.0)

            tick_idx += 1
            update_spectator(spectator, ego, cam, cfg.get("view","sensor"))

            # Route bookkeeping (keep it simple)
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
        try:
            set_sync(world, tm, FPS, False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
