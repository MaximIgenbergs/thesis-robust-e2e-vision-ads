#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector.py — CARLA expert data collector

- Async/sync runtime (configurable)
- Non-blocking image logging with counters
- Fixed-rate logging (Hz) or tick-based (every_n)
- CSV includes world & sensor timestamps/frames
- Live spectator view: "sensor" | "chase"
- Metadata dump (run_meta.json)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import yaml
import carla  # type: ignore

from agents_helpers import build_grp, pick_routes, next_high_level_command, make_agent
from sim_utils import (
    set_sync, attach_rgb, set_weather,
    update_spectator, destroy_actors, town_basename,
)
from io_utils import AsyncImageWriter, write_run_metadata

# ---------------- Constants ----------------

HOST: str = "localhost"
RPC_PORT: int = 2000
TM_PORT: int = 8000


# ---------------- Data classes ----------------

@dataclass
class CameraConfig:
    image_size_x: int
    image_size_y: int
    fov: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


# ---------------- Small helpers ----------------

def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "on", "y", "t"}
    return bool(x)


def _validate_config(cfg: Dict) -> None:
    """Light validation with clear errors; keep it simple."""
    required_top = ["town", "fps", "seed", "duration_s", "ego_blueprint",
                    "traffic_vehicles", "camera", "output_dir", "routes", "weather"]
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


# ---------------- Main ----------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CARLA expert data collector")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.config):
        print(f"[collector] Config not found: {args.config}", file=sys.stderr)
        return 2

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        _validate_config(cfg)
    except Exception as e:
        print(f"[collector] Bad config: {e}", file=sys.stderr)
        return 2

    # Runtime
    runtime_cfg = (cfg.get("runtime") or {})
    runtime_mode: str = str(runtime_cfg.get("mode", "sync")).lower()  # "sync" | "async"
    fast_mode: bool = _to_bool(runtime_cfg.get("fast_mode", False))    # only used in sync
    SYNC = (runtime_mode == "sync")

    # Logging
    log_cfg = (cfg.get("logging") or {})
    LOG_ON: bool = _to_bool(log_cfg.get("enabled", log_cfg.get("mode", True)))
    EVERY_N: int = int(log_cfg.get("every_n", 1))
    LOG_HZ: float = float(log_cfg.get("hz", 0.0))  # if >0, overrides every_n
    assert EVERY_N >= 1, "logging.every_n must be >= 1 when used"
    img_fmt: str = str(log_cfg.get("image_format", "png")).lower()  # png | jpg
    jpg_q: int = int(log_cfg.get("jpeg_quality", 90))
    png_cl: int = int(log_cfg.get("png_compress_level", 6))
    meta_dump: bool = _to_bool(log_cfg.get("metadata_dump", True))

    # View
    view_mode: str = str(cfg.get("view", "sensor")).lower()  # "sensor" | "chase"

    # Connect (do not hot-load the map immediately)
    client = carla.Client(HOST, RPC_PORT)
    client.set_timeout(20.0)
    world: "carla.World" = client.get_world()   # just attach to whatever is running

    # Reload only if needed (safer on 0.9.15)
    from sim_utils import town_basename
    current = town_basename(world.get_map().name)
    target = str(cfg["town"])

    if current != target:
        # small delay helps UE4 settle before reload
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

    # Spawns
    spawn_points = world.get_map().get_spawn_points()
    spawn_points.sort(key=lambda t: (t.location.x, t.location.y, t.location.z))

    # Cleanup tracking
    actors: List["carla.Actor"] = []
    meta_fp = None
    writer = None
    img_writer: Optional[AsyncImageWriter] = None

    # Graceful termination
    stop_flag = {"stop": False}
    def _handle_sig(signum, _frame):
        print(f"\n[collector] Signal {signum}; stopping at next tick.")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    try:
        # Ego
        ego_bp = world.get_blueprint_library().find(cfg["ego_blueprint"])
        ego = world.try_spawn_actor(ego_bp, spawn_points[0])
        if not ego:
            raise RuntimeError("Failed to spawn ego at spawn_points[0].")
        actors.append(ego)

        # Traffic
        veh_bps = list(world.get_blueprint_library().filter("vehicle.*"))
        i = 1
        while i < len(spawn_points) and len(actors) - 1 < int(cfg["traffic_vehicles"]):
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
            out_dir = os.path.join(cfg["output_dir"], f"{run_id}__{town_name}")
            img_dir = os.path.join(out_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            # CSV
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

            # background encoder with counters
            img_writer = AsyncImageWriter(
                max_queue=256,
                drop_if_full=True,
                image_format=img_fmt,
                jpeg_quality=jpg_q,
                png_compress_level=png_cl,
            )

            # Metadata dump
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
            print("[collector] logging disabled — no files will be written.")

        # RGB camera
        cam, q = attach_rgb(world, ego, cfg["camera"])
        actors.append(cam)

        # Spectator + ensure first frame exists
        spectator = world.get_spectator()
        for _ in range(120):
            if SYNC: world.tick()
            else:    world.wait_for_tick(seconds=1.0)
            update_spectator(spectator, ego, cam, view_mode)
            if len(q) > 0: break
        else:
            raise RuntimeError("Camera never produced a frame in warm-up")

        # Routes + agent
        grp = build_grp(world, sampling_resolution=2.0)
        rng = random.Random(SEED)
        routes = pick_routes(
            world, grp,
            int(cfg["routes"]["num_routes"]),
            float(cfg["routes"]["min_route_m"]),
            float(cfg["routes"]["max_route_m"]),
            rng,
        )
        agent = make_agent(world, ego)
        route_idx, step_in_route = 0, 0
        agent.set_destination(routes[route_idx][-1][0].transform.location)

        # Loop control
        tick_idx = 0
        snap0 = world.get_snapshot()
        end_time = snap0.timestamp.elapsed_seconds + float(cfg["duration_s"])

        # Pacing (sync & not fast)
        if SYNC and not fast_mode:
            dt_wall = 1.0 / FPS
            next_wall = time.perf_counter()

        # Fixed-rate logging clock (if hz > 0)
        next_log_ts = None
        if LOG_ON and LOG_HZ > 0.0:
            next_log_ts = snap0.timestamp.elapsed_seconds + (1.0 / LOG_HZ)

        # CSV flush cadence
        flush_mod = 500

        while True:
            if stop_flag["stop"]:
                break

            snap_now = world.get_snapshot()
            if snap_now and snap_now.timestamp.elapsed_seconds >= end_time:
                break

            # Planned command for label
            route = routes[route_idx]
            cmd = next_high_level_command(route, step_in_route)

            # Control
            control = agent.run_step()
            ego.apply_control(control)

            # Advance world
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

            # Spectator view
            update_spectator(spectator, ego, cam, view_mode)

            # Measurements
            tr = ego.get_transform()
            yaw = tr.rotation.yaw
            v = ego.get_velocity()
            speed = (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5

            # Decide if we should log this tick
            do_log = False
            if LOG_ON and writer:
                if LOG_HZ > 0.0:
                    if world_ts >= next_log_ts:
                        do_log = True
                        # advance schedule; catch up if behind
                        while next_log_ts <= world_ts:
                            next_log_ts += (1.0 / LOG_HZ)
                else:
                    do_log = (tick_idx % EVERY_N == 0)

            if do_log:
                img_path = ""
                sensor_frame = ""
                sensor_ts = ""
                if len(q) > 0:
                    img = q[-1]
                    sensor_frame = str(img.frame)
                    sensor_ts = f"{img.timestamp:.6f}"
                    if img_dir and img_writer:
                        ext = ".png" if img_fmt == "png" else ".jpg"
                        img_path = os.path.join(img_dir, f"{world_frame:09d}{ext}")
                        img_writer.submit(img_path, img)

                writer.writerow([
                    world_frame, f"{world_ts:.6f}",
                    sensor_frame, sensor_ts,
                    img_path,
                    f"{control.steer:.6f}", f"{control.throttle:.6f}", f"{control.brake:.6f}",
                    f"{speed:.6f}",
                    f"{tr.location.x:.3f}", f"{tr.location.y:.3f}", f"{tr.location.z:.3f}",
                    f"{yaw:.2f}", cmd, route_idx, step_in_route
                ])

                if (tick_idx % flush_mod) == 0:
                    try: meta_fp.flush()
                    except Exception: pass

            # Route bookkeeping
            tgt = routes[route_idx][-1][0].transform.location
            if tr.location.distance(tgt) < 5.0:
                route_idx = (route_idx + 1) % len(routes)
                agent.set_destination(routes[route_idx][-1][0].transform.location)
                step_in_route = 0
            else:
                step_in_route += 1

        if LOG_ON and out_dir:
            print(f"[collector] Finished. Wrote to: {out_dir}")
        else:
            print("[collector] Finished. No logging (disabled).")

    except Exception as e:
        print(f"[collector] Error: {e}", file=sys.stderr)
        return 1

    finally:
        # Close writer first (ensures all images are on disk)
        stats = None
        try:
            if img_writer:
                stats = img_writer.close(return_stats=True)
        except Exception:
            pass

        # Close CSV
        try:
            if meta_fp:
                meta_fp.flush()
                meta_fp.close()
        except Exception:
            pass

        # Stop sensors before destroy
        try:
            for a in actors:
                if a and a.type_id.startswith("sensor."):
                    try:
                        a.stop()
                    except Exception:
                        pass
        except Exception:
            pass

        destroy_actors(reversed(actors))

        # Leave server in async (safer default)
        try:
            set_sync(world, tm, int(cfg["fps"]), False)
        except Exception:
            pass

        if stats:
            submitted, written, dropped = stats["submitted"], stats["written"], stats["dropped"]
            print(f"[collector] Image I/O: submitted={submitted}, written={written}, dropped={dropped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
