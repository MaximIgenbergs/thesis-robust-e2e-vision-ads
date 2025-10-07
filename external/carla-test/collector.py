#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector.py
Data collection script for CARLA driving with a BehaviorAgent.

Responsibilities
---------------
- Load town, set deterministic seeds, and enable synchronous stepping.
- Spawn ego + background traffic; attach an RGB camera to the ego.
- Sample several short routes and drive them with a BehaviorAgent.
- Persist images (PNG) and a CSV of frame-aligned measurements.

Outputs
-------
<output_dir>/<YYYYmmdd_HHMMSS>__<town>/
  ├── images/                # <frame>.png (CARLA sensor.save_to_disk)
  └── frames.csv             # schema below

CSV schema (frames.csv)
-----------------------
frame, timestamp, image_path,
steer, throttle, brake,
speed_mps, x, y, z, yaw_deg,
command, route_idx, step_in_route

Usage
-----
$ python external/carla/collector.py --config config.yaml
(or place config.yaml next to this file and omit --config)

Assumptions
-----------
- CARLA server is running and reachable at HOST:RPC_PORT.
- Traffic Manager is reachable at TM_PORT.
- Config file provides keys read in `main()` (see code).

Notes
-----
- This script uses synchronous mode + fixed delta for reproducibility.
- Clean shutdown is handled via try/finally (actors destroyed; sync off).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import signal
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

import carla  # type: ignore

from agents_helpers import build_grp, pick_routes, next_high_level_command, make_agent


# -------- Configuration --------

HOST: str = "localhost"
RPC_PORT: int = 2000
TM_PORT: int = 8000


# -------- Data classes (optional clarity) --------

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


# -------- Utilities --------

def set_sync(world: "carla.World", tm: "carla.TrafficManager", fps: int, enabled: bool = True) -> None:
    """
    Enable/disable synchronous stepping for both World and Traffic Manager.

    Parameters
    ----------
    world : carla.World
        The CARLA world to configure.
    tm : carla.TrafficManager
        The Traffic Manager instance.
    fps : int
        Target frames per second when synchronous mode is enabled.
    enabled : bool, default True
        Whether to enable (True) or disable (False) sync mode.
    """
    settings = world.get_settings()
    settings.synchronous_mode = enabled
    settings.fixed_delta_seconds = (1.0 / fps) if enabled else None
    world.apply_settings(settings)
    tm.set_synchronous_mode(enabled)


def attach_rgb(world: "carla.World", ego: "carla.Actor", cam_cfg: Dict) -> Tuple["carla.Sensor", Deque]:
    """
    Attach an RGB camera to the ego and return the sensor + a single-item queue.

    Parameters
    ----------
    world : carla.World
        Active CARLA world.
    ego : carla.Actor
        Ego vehicle actor to attach the sensor to.
    cam_cfg : dict
        Camera parameters. Must contain keys:
        image_size_x, image_size_y, fov, x, y, z, roll, pitch, yaw.

    Returns
    -------
    (sensor, queue) : (carla.Sensor, collections.deque)
        The spawned camera sensor and a queue (maxlen=1) holding last frame.
    """
    required = {
        "image_size_x", "image_size_y", "fov",
        "x", "y", "z", "roll", "pitch", "yaw",
    }
    missing = required.difference(cam_cfg.keys())
    if missing:
        raise KeyError(f"Camera config missing keys: {sorted(missing)}")

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(int(cam_cfg["image_size_x"])))
    bp.set_attribute("image_size_y", str(int(cam_cfg["image_size_y"])))
    bp.set_attribute("fov", str(float(cam_cfg["fov"])))

    rel = carla.Transform(
        carla.Location(x=float(cam_cfg["x"]), y=float(cam_cfg["y"]), z=float(cam_cfg["z"])),
        carla.Rotation(
            roll=float(cam_cfg["roll"]),
            pitch=float(cam_cfg["pitch"]),
            yaw=float(cam_cfg["yaw"]),
        ),
    )
    cam: "carla.Sensor" = world.spawn_actor(bp, rel, attach_to=ego)
    q: Deque = deque(maxlen=1)

    # Keep only the latest frame to avoid backpressure in sync mode.
    cam.listen(lambda data: q.append(data))
    return cam, q


def set_weather(world: "carla.World", preset: str) -> None:
    """
    Set the world's weather to a named preset.

    Parameters
    ----------
    world : carla.World
        Active CARLA world.
    preset : str
        One of CLEAR_NOON, CLEAR_SUNSET, CLOUDY_NOON, WET_NOON, WET_CLOUDY_NOON,
        MID_RAINY_NOON, HARD_RAIN_NOON, SOFT_RAIN_NOON, CLEAR_NIGHT, CLOUDY_NIGHT.
        Defaults to ClearNoon if unknown.
    """
    W = carla.WeatherParameters
    table = {
        "CLEAR_NOON": W.ClearNoon,
        "CLEAR_SUNSET": W.ClearSunset,
        "CLOUDY_NOON": W.CloudyNoon,
        "WET_NOON": W.WetNoon,
        "WET_CLOUDY_NOON": W.WetCloudyNoon,
        "MID_RAINY_NOON": W.MidRainyNoon,
        "HARD_RAIN_NOON": W.HardRainNoon,
        "SOFT_RAIN_NOON": W.SoftRainNoon,
        "CLEAR_NIGHT": W.ClearNight,
        "CLOUDY_NIGHT": W.CloudyNight,
    }
    world.set_weather(table.get(preset, W.ClearNoon))


def _destroy_actors(actors: Iterable["carla.Actor"]) -> None:
    """Best-effort destruction of a collection of actors."""
    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass


# -------- Main --------

def main(argv: Optional[List[str]] = None) -> int:
    """
    Run a time-bounded collection session defined by a YAML config.

    Expected config.yaml keys
    -------------------------
    town : str
    seed : int
    fps : int
    weather.preset : str
    ego_blueprint : str                  # e.g., "vehicle.tesla.model3"
    traffic_vehicles : int
    camera : dict                        # see attach_rgb() for required keys
    routes.num_routes : int
    routes.min_route_m : float
    routes.max_route_m : float
    duration_s : float
    output_dir : str

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="CARLA data collector")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config (default: config.yaml)",
    )
    args = parser.parse_args(argv)

    # Load configuration
    if not os.path.isfile(args.config):
        print(f"[collector] Config not found: {args.config}", file=sys.stderr)
        return 2

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Connect to CARLA
    client = carla.Client(HOST, RPC_PORT)
    client.set_timeout(20.0)
    client.load_world(cfg["town"])
    world: "carla.World" = client.get_world()
    tm: "carla.TrafficManager" = client.get_trafficmanager(TM_PORT)

    # Determinism: seeds + sync
    SEED, FPS = int(cfg["seed"]), int(cfg["fps"])
    random.seed(SEED)
    np.random.seed(SEED)
    tm.set_random_device_seed(SEED)
    world.set_pedestrians_seed(SEED + 1)
    set_sync(world, tm, FPS, True)

    # Weather
    set_weather(world, cfg["weather"]["preset"])

    # Blueprint library and spawns
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    # Stable ordering so spawn_points[0] is deterministic across runs
    spawn_points.sort(key=lambda t: (t.location.x, t.location.y, t.location.z))

    # Actors to clean up in finally:
    actors: List["carla.Actor"] = []

    # Graceful termination on SIGINT/SIGTERM (sets a flag; cleanup still in finally)
    interrupted = {"stop": False}

    def _handle_sig(signum, frame):
        print(f"\n[collector] Received signal {signum}; stopping at next tick.")
        interrupted["stop"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    meta_fp = None
    cam = None

    try:
        # Ego vehicle
        ego_bp = bp_lib.find(cfg["ego_blueprint"])
        ego = world.try_spawn_actor(ego_bp, spawn_points[0])
        if not ego:
            raise RuntimeError("Failed to spawn ego at spawn_points[0].")
        actors.append(ego)

        # Traffic vehicles
        traffic: List["carla.Actor"] = []
        veh_bps = list(bp_lib.filter("vehicle.*"))
        i = 1
        while len(traffic) < int(cfg["traffic_vehicles"]) and i < len(spawn_points):
            bp = veh_bps[(len(traffic)) % len(veh_bps)]
            v = world.try_spawn_actor(bp, spawn_points[i])
            i += 1
            if v:
                v.set_autopilot(True, tm.get_port())
                traffic.append(v)
                actors.append(v)

        # Camera sensor
        cam, q = attach_rgb(world, ego, cfg["camera"])
        actors.append(cam)

        # Route planner + agent
        grp = build_grp(world, sampling_resolution=2.0)
        rng = random.Random(SEED)
        routes = pick_routes(
            world,
            grp,
            int(cfg["routes"]["num_routes"]),
            float(cfg["routes"]["min_route_m"]),
            float(cfg["routes"]["max_route_m"]),
            rng,
        )
        agent = make_agent(world, ego)

        # Output folders/files
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(cfg["output_dir"], f"{run_id}__{cfg['town']}")
        os.makedirs(out_dir, exist_ok=True)
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        meta_path = os.path.join(out_dir, "frames.csv")
        meta_fp = open(meta_path, "w", newline="")
        writer = csv.writer(meta_fp)
        writer.writerow(
            [
                "frame",
                "timestamp",
                "image_path",
                "steer",
                "throttle",
                "brake",
                "speed_mps",
                "x",
                "y",
                "z",
                "yaw_deg",
                "command",
                "route_idx",
                "step_in_route",
            ]
        )

        # Initialize route loop
        snapshot = world.get_snapshot()
        end_time = snapshot.timestamp.elapsed_seconds + float(cfg["duration_s"])
        route_idx = 0
        agent.set_destination(routes[route_idx][-1][0].transform.location)
        step_in_route = 0

        # Main loop (synchronous)
        while True:
            if interrupted["stop"]:
                break

            snap = world.get_snapshot()
            if snap.timestamp.elapsed_seconds >= end_time:
                break

            # Planned high-level command for logging
            route = routes[route_idx]
            cmd = next_high_level_command(route, step_in_route)

            # BehaviorAgent computes low-level control
            control = agent.run_step()
            ego.apply_control(control)

            world.tick()
            snap = world.get_snapshot()
            ts = snap.timestamp.elapsed_seconds
            frame = snap.frame

            # Latest image from queue (if any)
            img = q[-1] if q else None
            if img:
                img_path = os.path.join(img_dir, f"{frame:09d}.png")
                img.save_to_disk(img_path)
            else:
                img_path = ""

            # Measurements
            v = ego.get_velocity()
            speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
            tr = ego.get_transform()
            yaw = tr.rotation.yaw

            writer.writerow(
                [
                    frame,
                    f"{ts:.3f}",
                    img_path,
                    f"{control.steer:.6f}",
                    f"{control.throttle:.6f}",
                    f"{control.brake:.6f}",
                    f"{speed:.6f}",
                    f"{tr.location.x:.3f}",
                    f"{tr.location.y:.3f}",
                    f"{tr.location.z:.3f}",
                    f"{yaw:.2f}",
                    cmd,
                    route_idx,
                    step_in_route,
                ]
            )

            # Route bookkeeping: advance when close to target waypoint
            tgt = routes[route_idx][-1][0].transform.location
            if tr.location.distance(tgt) < 5.0:
                route_idx = (route_idx + 1) % len(routes)
                agent.set_destination(routes[route_idx][-1][0].transform.location)
                step_in_route = 0
            else:
                step_in_route += 1

        print(f"[collector] Finished. Wrote: {meta_path}")

    except Exception as e:
        print(f"[collector] Error: {e}", file=sys.stderr)
        return 1
    finally:
        # Stop sensor stream before destroy
        try:
            if cam is not None:
                cam.stop()
        except Exception:
            pass

        _destroy_actors(reversed(actors))  # destroy children before parents
        try:
            set_sync(world, tm, int(cfg["fps"]), False)
        except Exception:
            pass
        if meta_fp:
            try:
                meta_fp.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
