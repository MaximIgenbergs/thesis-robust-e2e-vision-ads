#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 drive_tcp.py — Run a TCP end-to-end model in CARLA (Lightning/TorchScript, no CLI flags)

 • Loads a TCP model **from config.yaml** (no command-line flags).
 • Supports **PyTorch Lightning** checkpoints (primary) and **TorchScript** .pt (optional).
 • Derives high-level commands from your route (via agents_helpers) and, if needed,
   provides a **target point** in ego coordinates to TCP variants that expect it.
 • Logging, spectator view, timing, and cleanup mirror collector.py.

Config schema additions (under top-level `tcp:`):

  tcp:
    ckpt: "~/Downloads/best_model.ckpt"     # path to .ckpt or .pt
    format: "lightning"                      # "lightning" | "torchscript"
    device: "cuda:0"                         # "cpu" or CUDA device
    input_w: 256                             # model input width
    input_h: 256                             # model input height
    n_cmds: 9                                # command vector length (if auto-infer fails)
    aim_dist: 4.0                            # meters ahead for target point
    model:
      class: "TCP.model.TCP"                # import path to model class (Lightning or plain nn.Module)
      kwargs: {}                             # kwargs for class or load_from_checkpoint
      config_yaml: null                      # optional: path to YAML to pass as 'config'
      config_class: "TCP.config:GlobalConfig"# optional: "module:Class" to instantiate as 'config'
      config_override: {}                    # optional: dict of overrides for config_class

Notes
-----
 • If your model returns waypoints instead of controls, a simple pure-pursuit-like
   controller converts them to (steer, throttle, brake) (tune gains later).
 • If the model forward signature is unusual, we **inspect it** and map inputs by
   parameter names (image/command/speed/target_point/measurements/etc.).
"""

from __future__ import annotations

import csv
import importlib
import os
import random
import signal
import sys
import time
import math
import inspect
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import carla  # type: ignore

import torch
from PIL import Image

from agents_helpers import build_grp, pick_routes, next_high_level_command
from sim_utils import (
    set_sync, attach_rgb, set_weather,
    update_spectator, destroy_actors, town_basename,
)
from io_utils import AsyncImageWriter, write_run_metadata

# ---------------- Constants ----------------
HOST: str = "localhost"
RPC_PORT: int = 2000
TM_PORT: int = 8000

# ---------------- Config validation ----------------

def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "on", "y", "t"}
    return bool(x)


def _validate_config(cfg: Dict) -> None:
    required_top = [
        "town", "fps", "seed", "duration_s", "camera", "output_dir", "routes", "weather"
    ]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"Missing config key: {k}")

    cam_req = ["x", "y", "z", "roll", "pitch", "yaw", "image_size_x", "image_size_y", "fov"]
    for k in cam_req:
        if k not in cfg["camera"]:
            raise ValueError(f"Missing camera config key: camera.{k}")

    routes_req = ["num_routes", "min_route_m", "max_route_m"]
    for k in routes_req:
        if k not in cfg["routes"]:
            raise ValueError(f"Missing routes config key: routes.{k}")

    if "preset" not in cfg["weather"]:
        raise ValueError("Missing weather.preset")

    if cfg["fps"] <= 0:
        raise ValueError("fps must be > 0")
    if cfg["duration_s"] <= 0:
        raise ValueError("duration_s must be > 0")
    if cfg["routes"]["num_routes"] <= 0:
        raise ValueError("routes.num_routes must be > 0")

# ---------------- Image utils ----------------

def carla_image_to_rgb_array(img: "carla.Image") -> np.ndarray:
    # CARLA returns BGRA uint8 buffer
    arr = np.frombuffer(img.raw_data, dtype=np.uint8)
    arr = arr.reshape((img.height, img.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]  # BGR -> RGB
    return rgb


def preprocess(img_arr: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    # size = (W,H)
    im = Image.fromarray(img_arr)
    im = im.resize(size, Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).float() / 255.0  # H,W,3
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return x

# ---------------- Route / target-point utils ----------------

def _closest_target_point_along_route(route, start_idx: int, aim_dist_m: float, ego_loc: "carla.Location") -> "carla.Location":
    """Pick a target point ~aim_dist_m ahead along the route from start_idx. Fallback to last waypoint."""
    if not route:
        return ego_loc
    last_loc = route[start_idx][0].transform.location if start_idx < len(route) else route[-1][0].transform.location
    dist = 0.0
    for i in range(start_idx + 1, len(route)):
        loc = route[i][0].transform.location
        step = loc.distance(last_loc)
        dist += step
        if dist >= aim_dist_m:
            return loc
        last_loc = loc
    return route[-1][0].transform.location


def _world_to_ego_xy(ego_tf: "carla.Transform", world_loc: "carla.Location") -> Tuple[float, float]:
    dx = world_loc.x - ego_tf.location.x
    dy = world_loc.y - ego_tf.location.y
    yaw = math.radians(ego_tf.rotation.yaw)
    cos, sin = math.cos(yaw), math.sin(yaw)
    x_fwd =  cos * dx + sin * dy
    y_right = -sin * dx + cos * dy
    return x_fwd, y_right

# ---------------- Command encoding ----------------

CMD_ORDER = ["FOLLOW", "LEFT", "RIGHT", "STRAIGHT"]  # base order


def encode_command(cmd: str, dim: int = 4) -> torch.Tensor:
    c = cmd.upper()
    key = "FOLLOW"
    if "LEFT" in c:
        key = "LEFT"
    elif "RIGHT" in c:
        key = "RIGHT"
    elif "STRAIGHT" in c or "LANEFOLLOW" in c:
        key = "STRAIGHT" if "STRAIGHT" in c else "FOLLOW"
    onehot = torch.zeros(1, dim, dtype=torch.float32)
    idx = CMD_ORDER.index(key) if key in CMD_ORDER else 0
    if idx < dim:
        onehot[0, idx] = 1.0
    return onehot


def encode_command3(cmd: str) -> torch.Tensor:
    c = cmd.upper()
    idx = 2  # STRAIGHT default
    if "LEFT" in c:
        idx = 0
    elif "RIGHT" in c:
        idx = 1
    v = torch.zeros(1, 3, dtype=torch.float32)
    v[0, idx] = 1.0
    return v

# ---------------- TCP model wrapper ----------------

class TCPWrapper:
    """Wrapper that adapts multiple TCP signatures (with/without target_point)."""
    def __init__(self, *, fmt: str, ckpt_path: str, model_class_path: Optional[str], device: str = "cpu", model_kwargs: Optional[dict] = None, n_cmds: Optional[int] = None):
        self.fmt = fmt
        self.device = torch.device(device)
        self.model = None
        self.input_size = (320, 160)  # (W,H) default; overridden from cfg
        self.model_kwargs = model_kwargs or {}
        self.cmd_dim = n_cmds  # may be None; attempt to infer
        self.fwd_sig = None
        self._dbg_printed = False

        if fmt == "lightning":
            if not model_class_path:
                raise ValueError("tcp.model.class must be provided for fmt=lightning")
            module, cls_name = model_class_path.rsplit(".", 1)
            mod = importlib.import_module(module)
            cls = getattr(mod, cls_name)

            def _safe_load_ckpt(path: str):
                try:
                    return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
                except Exception:
                    print("[drive_tcp] Warning: falling back to torch.load(weights_only=False). Only load trusted checkpoints.", file=sys.stderr)
                    return torch.load(path, map_location="cpu")

            if hasattr(cls, "load_from_checkpoint"):
                try:
                    self.model = cls.load_from_checkpoint(ckpt_path, map_location=self.device, **self.model_kwargs)
                except ModuleNotFoundError:
                    obj = _safe_load_ckpt(ckpt_path)
                    state = obj.get("state_dict", obj)
                    m = cls(**self.model_kwargs)
                    m.load_state_dict(state, strict=False)
                    self.model = m
            else:
                obj = _safe_load_ckpt(ckpt_path)
                state = obj.get("state_dict", obj)
                # strip common prefixes
                def _strip(sd):
                    out = {}
                    for k, v in sd.items():
                        nk = k
                        for pref in ("model.", "net.", "module."):
                            if nk.startswith(pref):
                                nk = nk[len(pref):]
                                break
                        out[nk] = v
                    return out
                if isinstance(state, dict):
                    state = _strip(state)
                m = cls(**self.model_kwargs)
                m.load_state_dict(state, strict=False)
                self.model = m

            self.model.to(self.device).eval()
            try:
                self.fwd_sig = inspect.signature(self.model.forward)
            except Exception:
                self.fwd_sig = None
            if self.cmd_dim is None:
                self.cmd_dim = self._infer_cmd_dim(self.model)

        elif fmt == "torchscript":
            self.model = torch.jit.load(ckpt_path, map_location=self.device)
            self.model.eval()
            try:
                self.fwd_sig = inspect.signature(self.model.forward)
            except Exception:
                self.fwd_sig = None
            if self.cmd_dim is None:
                self.cmd_dim = 4
        else:
            raise ValueError("tcp.format must be 'lightning' or 'torchscript'")

    def _infer_cmd_dim(self, model) -> int:
        candidates = set()
        for _, p in getattr(model, 'named_parameters', lambda: [])():
            if p.ndim == 2:
                d = p.shape[1]
                if d in (4, 5, 6, 8, 9, 10):
                    candidates.add(d)
        if candidates:
            for pref in (9, 8, 6, 5, 4, 10):
                if pref in candidates:
                    return pref
            return max(candidates)
        return 4

    def set_input_size(self, w: int, h: int):
        self.input_size = (w, h)

    @torch.no_grad()
    def forward(self, img_arr: np.ndarray, cmd_str: str, speed: float,
                target_point_xy: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float]:
        import inspect
        # --- inputs ---
        x = preprocess(img_arr, self.input_size).to(self.device)
        # ensure we always have a target point (fallback straight ahead)
        if target_point_xy is None:
            # 4 m ahead by default
            target_point_xy = (4.0, 0.0)
        tp = torch.tensor([[float(target_point_xy[0]), float(target_point_xy[1])]],
                        dtype=x.dtype, device=self.device)
        # prepare also common aux (some TCPs ignore them; harmless to compute)
        cmd_full = encode_command(cmd_str, dim=self.cmd_dim or 4).to(self.device)
        cmd3 = encode_command3(cmd_str).to(self.device)
        spd = torch.tensor([[speed]], dtype=x.dtype, device=self.device)
        meas4 = torch.cat([spd, cmd3], dim=1)

        # --- inspect signature ---
        try:
            sig = inspect.signature(self.model.forward)
            params = list(sig.parameters.items())[1:]  # skip self
        except Exception:
            params = []
        kinds = [(n, p.kind) for n, p in params]
        if not getattr(self, "_dbg_printed", False):
            print("[drive_tcp] forward param kinds:", [(n, str(k)) for n, k in kinds])
            self._dbg_printed = True

        # map values by name
        def val_for(name: str):
            key = name.replace("_", "").lower()
            if key in ("state","image","img","x","rgb","inputs"): return x
            if key in ("targetpoint","target","tp","goal","waypoint","wp","routepoint"): return tp
            if key in ("command","cmd","cmdfull","cmd9","routecommand","highlevel"): return cmd_full
            if key in ("command3","cmd3"): return cmd3
            if key in ("speed","vel","v"): return spd
            if key in ("measurements","meas","meas4"): return meas4
            # heuristics
            if "point" in key or "goal" in key or "wp" in key: return tp
            if "meas" in key: return meas4
            if "cmd" in key: return cmd_full
            if "spd" in key or "vel" in key: return spd
            return x

        # build positional/keyword respecting kinds
        pos_args, kw_args = [], {}
        for n, k in kinds:
            v = val_for(n)
            if k in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                pos_args.append(v)
            elif k is inspect.Parameter.KEYWORD_ONLY:
                kw_args[n] = v
            else:  # VAR_POSITIONAL/VAR_KEYWORD
                kw_args[n] = v
        if not hasattr(self, "_dbg_called"):
            print("[drive_tcp] mapped forward call:", ("pos", [tuple(a.shape) for a in pos_args]),
                ("kw", {n: tuple(t.shape) for n,t in kw_args.items()}))
            self._dbg_called = True

        # helper to normalize outputs
        def to_controls(out_tensor: torch.Tensor) -> Tuple[float,float,float]:
            if isinstance(out_tensor, (list, tuple)): out_tensor = out_tensor[0]
            y = out_tensor.detach().cpu()
            if y.ndim == 3 and y.shape[-1] == 2:
                # waypoint controller
                wp = y[0, 0]
                x_wp, y_wp = float(wp[0]), float(wp[1])
                steer = float(np.clip(math.atan2(y_wp, max(1e-3, x_wp)) / 0.4, -1.0, 1.0))
                target_v = 6.0
                v_err = target_v - float(spd[0,0].cpu())
                throttle = float(np.clip(0.2 * v_err, 0.0, 1.0))
                brake = 0.0 if v_err >= 0 else float(np.clip(-0.3 * v_err, 0.0, 1.0))
                return steer, throttle, brake
            y = y.squeeze(0).float().numpy()
            return float(y[0]), float(y[1]), float(y[2])

        # attempt 1: unbound forward (bypass Lightning __call__)
        try:
            unbound = type(self.model).forward
        except Exception:
            unbound = None
        if unbound is not None:
            try:
                return to_controls(unbound(self.model, *pos_args, **kw_args))
            except TypeError:
                pass
            # try strict 2-positional (state, target_point)
            try:
                return to_controls(unbound(self.model, x, tp))
            except TypeError:
                pass

        # attempt 2: bound forward with kwargs/kinds
        try:
            return to_controls(self.model.forward(*pos_args, **kw_args))
        except TypeError:
            pass
        try:
            return to_controls(self.model.forward(x, tp))
        except TypeError:
            pass

        # attempt 3: module __call__
        try:
            return to_controls(self.model(*pos_args, **kw_args))
        except TypeError as e:
            # last resort: wrap with a tiny adapter to force a (state, target_point) signature
            class _Adapter(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner
                def forward(self, state, target_point):
                    # try variants
                    for fn in (self.inner.forward, getattr(self.inner, "__call__", None)):
                        if fn is None: continue
                        for args in ((state, target_point),):
                            try: return fn(*args)
                            except TypeError: pass
                        try: return fn(state=state, target_point=target_point)
                        except TypeError: pass
                    raise TypeError("adapter could not satisfy inner model forward")
            try:
                adapted = _Adapter(self.model).to(self.device).eval()
                return to_controls(adapted(x, tp))
            except Exception as e2:
                raise RuntimeError(f"TCP forward could not be satisfied: {e2}") from e
            
            # --- after the __call__ try; before raising
    # 4) Try TCP forward could not be satisfied by tensor-args or dict-batch calls.")



# ---------------- Helpers to load TCP from cfg ----------------

def _load_tcp_from_cfg(cfg: Dict) -> Tuple[TCPWrapper, Dict]:
    tcp_cfg = cfg.get("tcp") or {}
    ckpt = os.path.expanduser(tcp_cfg.get("ckpt", "~/Downloads/best_model.ckpt"))
    fmt = str(tcp_cfg.get("format", "lightning")).lower()
    device = str(tcp_cfg.get("device", "cuda:0"))
    input_w = int(tcp_cfg.get("input_w", 256))
    input_h = int(tcp_cfg.get("input_h", 256))
    n_cmds = tcp_cfg.get("n_cmds")
    aim_dist = float(tcp_cfg.get("aim_dist", 4.0))

    model_block = tcp_cfg.get("model") or {}
    model_class = model_block.get("class", "TCP.model.TCP")
    kwargs = model_block.get("kwargs") or {}

    # Optional config providers
    cfg_yaml = model_block.get("config_yaml")
    cfg_class = model_block.get("config_class")
    cfg_override = model_block.get("config_override") or {}

    # Build kwargs['config'] if provided
    if cfg_yaml:
        with open(os.path.expanduser(cfg_yaml), 'r') as f:
            kwargs.setdefault('config', yaml.safe_load(f))
    if cfg_class:
        mod_path, cls_name = cfg_class.split(":", 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        cfg_obj = cls(**cfg_override)
        kwargs['config'] = cfg_obj

    tcp = TCPWrapper(
        fmt=fmt,
        ckpt_path=ckpt,
        model_class_path=model_class if fmt == "lightning" else None,
        device=device,
        model_kwargs=kwargs,
        n_cmds=int(n_cmds) if n_cmds is not None else None,
    )
    tcp.set_input_size(input_w, input_h)
    # return wrapper and a small dict with runtime hints
    return tcp, {"aim_dist": aim_dist, "config_obj": kwargs.get('config')}

# ---------------- Main ----------------

def main() -> int:
    # No CLI: read config path from env or default to ./config.yaml
    cfg_path = os.environ.get("TCP_CONFIG", "config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"[drive_tcp] Config not found: {cfg_path}", file=sys.stderr)
        return 2

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    try:
        _validate_config(cfg)
    except Exception as e:
        print(f"[drive_tcp] Bad config: {e}", file=sys.stderr)
        return 2

    # Runtime
    runtime_cfg = (cfg.get("runtime") or {})
    runtime_mode: str = str(runtime_cfg.get("mode", "sync")).lower()
    fast_mode: bool = _to_bool(runtime_cfg.get("fast_mode", False))
    SYNC = (runtime_mode == "sync")

    # Logging
    log_cfg = (cfg.get("logging") or {})
    LOG_ON: bool = _to_bool(log_cfg.get("enabled", log_cfg.get("mode", True)))
    EVERY_N: int = int(log_cfg.get("every_n", 1))
    LOG_HZ: float = float(log_cfg.get("hz", 0.0))
    assert EVERY_N >= 1, "logging.every_n must be >= 1 when used"
    img_fmt: str = str(log_cfg.get("image_format", "png")).lower()  # png | jpg
    jpg_q: int = int(log_cfg.get("jpeg_quality", 90))
    png_cl: int = int(log_cfg.get("png_compress_level", 6))
    meta_dump: bool = _to_bool(log_cfg.get("metadata_dump", True))

    view_mode: str = str(cfg.get("view", "sensor")).lower()  # "sensor" | "chase"

    # Connect to CARLA
    client = carla.Client(HOST, RPC_PORT)
    client.set_timeout(20.0)
    world: "carla.World" = client.get_world()

    # Load map if not already loaded
    current = town_basename(world.get_map().name)
    target = str(cfg["town"])
    if current != target:
        time.sleep(1.0)
        client.load_world(target)
        world = client.get_world()

    tm: "carla.TrafficManager" = client.get_trafficmanager(TM_PORT)

    # Seeds + mode
    SEED, FPS = int(cfg["seed"]), int(cfg["fps"])
    random.seed(SEED)
    np.random.seed(SEED)
    tm.set_random_device_seed(SEED)
    world.set_pedestrians_seed(SEED + 1)
    set_sync(world, tm, FPS, SYNC)

    # Warm-up
    if SYNC:
        for _ in range(30):
            world.tick()
    else:
        for _ in range(30):
            world.wait_for_tick(seconds=1.0)

    # Weather
    set_weather(world, cfg["weather"]["preset"])

    # Spawn ego only (no autopilot here)
    spawn_points = world.get_map().get_spawn_points()
    spawn_points.sort(key=lambda t: (t.location.x, t.location.y, t.location.z))

    actors: List["carla.Actor"] = []
    meta_fp = None
    writer = None
    img_writer: Optional[AsyncImageWriter] = None

    # Graceful termination
    stop_flag = {"stop": False}
    def _handle_sig(signum, _frame):
        print(f"[drive_tcp] Signal {signum}; stopping at next tick.")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    # Instantiate model from cfg
    try:
        tcp, tcp_hints = _load_tcp_from_cfg(cfg)
    except Exception as e:
        print(f"[drive_tcp] Model load error: {e}", file=sys.stderr)
        return 2

    aim_dist = float(tcp_hints.get("aim_dist", 4.0))
    cfg_obj = tcp_hints.get("config_obj")

    try:
        ego_bp_name = cfg.get("ego_blueprint", "vehicle.tesla.model3")
        ego_bp = world.get_blueprint_library().find(ego_bp_name)
        ego = world.try_spawn_actor(ego_bp, spawn_points[0])
        if not ego:
            raise RuntimeError("Failed to spawn ego at spawn_points[0].")
        actors.append(ego)

        # Optional background traffic
        n_traffic = int(cfg.get("traffic_vehicles", 0))
        if n_traffic > 0:
            veh_bps = list(world.get_blueprint_library().filter("vehicle.*"))
            i = 1
            while i < len(spawn_points) and len(actors) - 1 < n_traffic:
                bp = veh_bps[(len(actors) - 1) % len(veh_bps)]
                v = world.try_spawn_actor(bp, spawn_points[i]); i += 1
                if v:
                    v.set_autopilot(True, tm.get_port())
                    actors.append(v)

        # OUTPUTS
        out_dir = None
        img_dir = None
        if LOG_ON:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                town_name = town_basename(world.get_map().name)
            except Exception:
                town_name = cfg.get("town", "UnknownTown")
            out_dir = os.path.join(cfg["output_dir"], f"{run_id}__{town_name}__tcp")
            img_dir = os.path.join(out_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            meta_path = os.path.join(out_dir, "frames.csv")
            meta_fp = open(meta_path, "w", newline="")
            writer = csv.writer(meta_fp)
            writer.writerow([
                "world_frame","world_ts",
                "sensor_frame","sensor_ts",
                "image_path",
                "steer","throttle","brake",
                "speed_mps","x","y","z","yaw_deg",
                "command","route_idx","step_in_route"
            ])

            img_writer = AsyncImageWriter(
                max_queue=256,
                drop_if_full=True,
                image_format=img_fmt,
                jpeg_quality=jpg_q,
                png_compress_level=png_cl,
            )

            if meta_dump:
                write_run_metadata(
                    out_dir=out_dir,
                    world=world,
                    tm_port=TM_PORT,
                    cfg=cfg,
                    run_id=run_id,
                    host=HOST,
                    rpc_port=RPC_PORT,
                )
        else:
            print("[drive_tcp] logging disabled — no files will be written.")

        # RGB camera
        cam, q = attach_rgb(world, ego, cfg["camera"])  # q: collections.deque[carla.Image]
        actors.append(cam)

        # Spectator warm-up
        spectator = world.get_spectator()
        for _ in range(120):
            if SYNC: world.tick()
            else:    world.wait_for_tick(seconds=1.0)
            update_spectator(spectator, ego, cam, view_mode)
            if len(q) > 0: break
        else:
            raise RuntimeError("Camera never produced a frame in warm-up")

        # Routes & command planning
        grp = build_grp(world, sampling_resolution=2.0)
        rng = random.Random(SEED)
        routes = pick_routes(
            world, grp,
            int(cfg["routes"]["num_routes"]),
            float(cfg["routes"]["min_route_m"]),
            float(cfg["routes"]["max_route_m"]),
            rng,
        )
        route_idx, step_in_route = 0, 0

        # Loop control
        tick_idx = 0
        snap0 = world.get_snapshot()
        end_time = snap0.timestamp.elapsed_seconds + float(cfg["duration_s"])

        if SYNC and not fast_mode:
            dt_wall = 1.0 / FPS
            next_wall = time.perf_counter()

        next_log_ts = None
        if LOG_ON and LOG_HZ > 0.0:
            next_log_ts = snap0.timestamp.elapsed_seconds + (1.0 / LOG_HZ)

        flush_mod = 500

        while True:
            if stop_flag["stop"]:
                break

            snap_now = world.get_snapshot()
            if snap_now and snap_now.timestamp.elapsed_seconds >= end_time:
                break

            route = routes[route_idx]
            cmd_str = next_high_level_command(route, step_in_route)

            img = q[-1] if len(q) > 0 else None
            if img is None:
                if SYNC:
                    if not fast_mode:
                        next_wall += dt_wall
                        sleep = next_wall - time.perf_counter()
                        if sleep > 0:
                            time.sleep(sleep)
                        else:
                            next_wall = time.perf_counter()
                    world.tick()
                else:
                    world.wait_for_tick(seconds=1.0)
                continue

            tr = ego.get_transform()
            vel = ego.get_velocity()
            speed = float((vel.x*vel.x + vel.y*vel.y + vel.z*vel.z) ** 0.5)

            img_arr = carla_image_to_rgb_array(img)

            # Build a target point ~aim_dist ahead in ego frame
            aim = aim_dist
            if hasattr(cfg_obj, 'aim_dist'):
                try: aim = float(cfg_obj.aim_dist)
                except Exception: pass
            try:
                tgt_world = _closest_target_point_along_route(route, step_in_route, aim, tr.location)
                tp_xy = _world_to_ego_xy(tr, tgt_world)
            except Exception:
                tp_xy = None
            # Hard fallback: if we failed to compute a route-based target point, aim straight ahead
            if tp_xy is None:
                tp_xy = (aim, 0.0)

            try:
                steer, throttle, brake = tcp.forward(img_arr, cmd_str, speed, tp_xy)
            except Exception as e:
                print(f"[drive_tcp] Inference error: {e}", file=sys.stderr)
                steer, throttle, brake = 0.0, 0.0, 1.0

            control = carla.VehicleControl()
            control.steer = float(np.clip(steer, -1.0, 1.0))
            control.throttle = float(np.clip(throttle, 0.0, 1.0))
            control.brake = float(np.clip(brake, 0.0, 1.0))
            ego.apply_control(control)

            if SYNC:
                if not fast_mode:
                    next_wall += dt_wall
                    sleep = next_wall - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        next_wall = time.perf_counter()
                world.tick()
                snap = world.get_snapshot()
            else:
                snap = world.wait_for_tick(seconds=1.0)
                if snap is None:
                    continue

            tick_idx += 1
            world_ts = snap.timestamp.elapsed_seconds
            world_frame = snap.frame

            update_spectator(spectator, ego, cam, view_mode)

            do_log = False
            if LOG_ON and writer:
                if LOG_HZ > 0.0:
                    if world_ts >= next_log_ts:
                        do_log = True
                        while next_log_ts <= world_ts:
                            next_log_ts += (1.0 / LOG_HZ)
                else:
                    do_log = (tick_idx % EVERY_N == 0)

            if do_log:
                img_path = ""
                if img is not None and img_dir and img_writer:
                    ext = ".png" if img_fmt == "png" else ".jpg"
                    img_path = os.path.join(img_dir, f"{world_frame:09d}{ext}")
                    img_writer.submit(img_path, img)

                yaw = tr.rotation.yaw
                if writer:
                    writer.writerow([
                        world_frame, f"{world_ts:.6f}",
                        str(img.frame) if img else "", f"{getattr(img, 'timestamp', 0.0):.6f}" if img else "",
                        img_path,
                        f"{control.steer:.6f}", f"{control.throttle:.6f}", f"{control.brake:.6f}",
                        f"{speed:.6f}",
                        f"{tr.location.x:.3f}", f"{tr.location.y:.3f}", f"{tr.location.z:.3f}",
                        f"{yaw:.2f}", cmd_str, route_idx, step_in_route
                    ])
                    if (tick_idx % flush_mod) == 0:
                        try: meta_fp.flush()
                        except Exception: pass

            tgt = routes[route_idx][-1][0].transform.location
            if tr.location.distance(tgt) < 5.0:
                route_idx = (route_idx + 1) % len(routes)
                step_in_route = 0
            else:
                step_in_route += 1

        if LOG_ON and out_dir:
            print(f"[drive_tcp] Finished. Wrote to: {out_dir}")
        else:
            print("[drive_tcp] Finished. No logging (disabled).")

    except Exception as e:
        print(f"[drive_tcp] Error: {e}", file=sys.stderr)
        return 1

    finally:
        stats = None
        try:
            if img_writer:
                stats = img_writer.close(return_stats=True)
        except Exception:
            pass
        try:
            if meta_fp:
                meta_fp.flush(); meta_fp.close()
        except Exception:
            pass
        try:
            for a in actors:
                if a and a.type_id.startswith("sensor."):
                    try: a.stop()
                    except Exception: pass
        except Exception:
            pass
        destroy_actors(reversed(actors))
        try:
            set_sync(world, tm, int(cfg["fps"]), False)
        except Exception:
            pass
        if stats:
            submitted, written, dropped = stats["submitted"], stats["written"], stats["dropped"]
            print(f"[drive_tcp] Image I/O: submitted={submitted}, written={written}, dropped={dropped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
