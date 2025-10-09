# -*- coding: utf-8 -*-
"""
sim_utils.py — simulator helpers (kept import-light).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import carla  # type: ignore


# ---------- Sync / world settings ----------

def set_sync(world: "carla.World", tm: "carla.TrafficManager", fps: int, enabled: bool) -> None:
    settings = world.get_settings()
    settings.synchronous_mode = enabled
    settings.fixed_delta_seconds = (1.0 / fps) if enabled else None
    try:
        settings.max_substeps = 2
        settings.max_substep_delta_time = (1.0 / max(fps, 1)) / 2.0
    except Exception:
        pass
    world.apply_settings(settings)
    tm.set_synchronous_mode(enabled)


# ---------- Sensors & weather ----------

def attach_rgb(world: "carla.World", ego: "carla.Actor", cam_cfg: dict):
    required = {"image_size_x", "image_size_y", "fov", "x", "y", "z", "roll", "pitch", "yaw"}
    missing = required.difference(cam_cfg.keys())
    if missing:
        raise KeyError(f"Camera config missing keys: {sorted(missing)}")

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(int(cam_cfg["image_size_x"])))
    bp.set_attribute("image_size_y", str(int(cam_cfg["image_size_y"])))
    bp.set_attribute("fov", str(float(cam_cfg["fov"])))

    rel = carla.Transform(
        carla.Location(x=float(cam_cfg["x"]), y=float(cam_cfg["y"]), z=float(cam_cfg["z"])),
        carla.Rotation(roll=float(cam_cfg["roll"]), pitch=float(cam_cfg["pitch"]), yaw=float(cam_cfg["yaw"])),
    )
    cam: "carla.Sensor" = world.spawn_actor(bp, rel, attach_to=ego)

    from collections import deque
    q = deque(maxlen=1)
    cam.listen(lambda data: q.append(data))
    return cam, q


def set_weather(world: "carla.World", preset: str) -> None:
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


def destroy_actors(actors: Iterable["carla.Actor"]) -> None:
    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass


# ---------- Spectator views ----------

def _sensor_view_transform(cam: "carla.Sensor") -> "carla.Transform":
    return cam.get_transform()


def _chase_view_transform(ego_tf: "carla.Transform") -> "carla.Transform":
    BACK, UP, RIGHT, PITCH = 6.0, 2.5, 0.0, -10.0
    fwd = ego_tf.get_forward_vector()
    rgt = ego_tf.get_right_vector()
    loc = carla.Location(
        x=ego_tf.location.x - fwd.x * BACK + rgt.x * RIGHT,
        y=ego_tf.location.y - fwd.y * BACK + rgt.y * RIGHT,
        z=ego_tf.location.z - fwd.z * BACK + UP,
    )
    rot = carla.Rotation(pitch=PITCH, yaw=ego_tf.rotation.yaw, roll=0.0)
    return carla.Transform(loc, rot)


def update_spectator(spectator: "carla.Actor", ego: "carla.Vehicle", cam: "carla.Sensor", view_mode: str) -> None:
    try:
        if view_mode == "chase":
            spectator.set_transform(_chase_view_transform(ego.get_transform()))
        else:
            spectator.set_transform(_sensor_view_transform(cam))
    except RuntimeError:
        pass


# ---------- Misc ----------

def town_basename(name: str) -> str:
    """Extract TownXX from 'Carla/Maps/TownXX'."""
    import os
    return os.path.basename(name).split(".")[0]
