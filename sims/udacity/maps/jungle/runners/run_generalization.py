"""
Generalization run (no perturbations).
- Spawns at a specific waypoint, then drives using the selected model.
- Uses the same logging pipeline as robustness runs.
"""

from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Union
import numpy as np

# add project root & perturbation-drive to path (for consistent git snapshot only)
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PD_DIR = ROOT / "external" / "perturbation-drive"
if str(PD_DIR) not in sys.path:
    sys.path.insert(0, str(PD_DIR))

from external.udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from external.udacity_gym.agent_callback import PDPreviewCallback
from external.udacity_gym.logger import ScenarioOutcomeWriter, ScenarioOutcomeLite, ScenarioLite

from sims.udacity.maps.configs.run import HOST, PORT
from sims.udacity.maps.jungle.configs import paths, run
from sims.udacity.logging.eval_runs import (
    RunLogger, prepare_run_dir, module_public_dict, best_effort_git_sha, pip_freeze
)
from sims.udacity.adapters.dave2_adapter import Dave2Adapter
from sims.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter

START_WAYPOINT = 200
MAP_NAME = "jungle_reverse"

def _abs(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _build_adapter(model_name: str, image_size_hw, ckpt_path: Path | None):
    if model_name == "dave2":
        return Dave2Adapter(weights=ckpt_path, image_size_hw=image_size_hw, device=None, normalize="imagenet")
    elif model_name == "dave2_gru":
        return Dave2GRUAdapter(weights=ckpt_path, image_size_hw=image_size_hw, device=None, normalize="imagenet")
    raise ValueError(f"Unknown MODEL_NAME '{model_name}' in sims/udacity/configs/jungle/run.py")


def _force_start_episode(env: UdacityGym, track=MAP_NAME, weather="sunny", daytime="day", timeout_s=30.0):
    """Enter the map and wait until frames arrive (no Unity restart)."""
    env.reset(track=track, weather=weather, daytime=daytime)
    t0 = time.perf_counter()
    last_emit = 0.0
    while True:
        obs = env.observe()
        if obs and obs.input_image is not None:
            return
        now = time.perf_counter()
        if now - last_emit > 1.0:
            try:
                env.simulator.sim_executor.send_track(track, weather, daytime)
            except Exception:
                pass
            last_emit = now
        if now - t0 > timeout_s:
            raise RuntimeError(f"Failed to enter {track} before timeout.")
        time.sleep(0.2)

def _await_scene_ready(env: UdacityGym, min_frames: int = 8, max_wait_s: float = 5.0) -> None:
    t0 = time.perf_counter()
    last_t = None
    got = 0
    while got < min_frames and (time.perf_counter() - t0) < max_wait_s:
        obs = env.observe()
        if obs and obs.input_image is not None:
            if last_t is None or obs.time != last_t:
                got += 1
                last_t = obs.time
        time.sleep(0.02)


def _metrics(env: UdacityGym):
    try:
        m = env.simulator.get_episode_metrics() or {}
    except Exception:
        m = {}
    return {
        "outOfTrackCount": int(m.get("outOfTrackCount", 0)),
        "collisionCount": int(m.get("collisionCount", 0)),
    }


def _settle_metrics(env: UdacityGym, prev: dict, settle_seconds: float = 1.5, probe_hz: float = 10.0) -> dict:
    """Let Unity flush counters, then return delta vs. previous snapshot."""
    import time as _t
    delay = 1.0 / max(probe_hz, 1.0)
    deadline = _t.perf_counter() + max(settle_seconds, 0.0)
    last = _metrics(env)
    while _t.perf_counter() < deadline:
        _t.sleep(delay)
        cur = _metrics(env)
        if cur == last:
            break
        last = cur
    return {
        "outOfTrackCount": max(0, last["outOfTrackCount"] - prev["outOfTrackCount"]),
        "collisionCount":  max(0, last["collisionCount"]   - prev["collisionCount"]),
    }

def _spawn_waypoint(env: UdacityGym, idx: int) -> None:
    # Make sure Unity is already streaming frames (avoids overwrites)
    _await_scene_ready(env, min_frames=8, max_wait_s=5.0)

    print(f"[spawn] requesting waypoint {idx} (queued to executor)")
    env.simulator.sim_executor.request_spawn_waypoint(int(idx))

    # give the two emits a moment to go out and Unity to apply them
    time.sleep(0.6)
    print(f"[spawn] request for waypoint {idx} dispatched (executor emitted twice).")


def _predict_to_action(adapter, img: np.ndarray) -> UdacityAction:
    # Adapter may be call-able or expose .predict
    try:
        out = getattr(adapter, "predict")(img)
    except AttributeError:
        out = adapter(img)

    if isinstance(out, UdacityAction):
        return out
    if isinstance(out, dict):
        steer = out.get("steer", out.get("steering", out.get("steering_angle"), 0.0))
        thr   = out.get("throttle", out.get("accel", out.get("throttle_cmd"), 0.0))
        return UdacityAction(
            steering_angle=float(np.clip(float(steer), -1.0, 1.0)),
            throttle=float(np.clip(float(thr), -1.0, 1.0)),
        )
    arr = np.array(out).reshape(-1)
    return UdacityAction(
        steering_angle=float(np.clip(float(arr[0]), -1.0, 1.0)),
        throttle=float(np.clip(float(arr[1]), -1.0, 1.0)),
    )


def _run_episode(env: UdacityGym,
                 adapter,
                 preview: PDPreviewCallback,
                 max_steps: int,
                 save_images: bool) -> ScenarioOutcomeLite:

    _force_start_episode(env,
                         track=MAP_NAME,
                         weather=getattr(run, "WEATHER", "sunny"),
                         daytime=getattr(run, "DAYTIME", "day"))
    
    _spawn_waypoint(env, START_WAYPOINT)

    metrics_base = _metrics(env)

    # per-episode buffers
    frames, xte, speeds, actions, pos = [], [], [], [], []
    images = [] if save_images else None

    offtrack_count = 0
    collision_count = 0

    step = 0
    t0 = time.perf_counter()
    obs = env.observe()

    try:
        while step < max_steps:
            if obs is None or obs.input_image is None:
                time.sleep(0.005)
                obs = env.observe()
                continue

            img_np = np.asarray(obs.input_image, dtype=np.uint8).copy()
            if save_images:
                images.append(img_np)

            action = _predict_to_action(adapter, img_np)

            # Preview (show raw)
            preview(obs, display_image_np=img_np, action=action, perturbation="none")

            last_time = obs.time
            obs, reward, terminated, truncated, info = env.step(action)

            # events
            evs = (info or {}).get("events") or []
            for ev in evs:
                k = (ev.get("key") or ev.get("type") or "").lower()
                if k == "out_of_track":
                    offtrack_count += 1
                elif k == "collision":
                    collision_count += 1
            if (info or {}).get("out_of_track") is True:
                offtrack_count += 1
            if (info or {}).get("collision") is True:
                collision_count += 1

            # logs
            frames.append(step)
            xte.append(float(getattr(obs, "cte", 0.0)))
            speeds.append(float(getattr(obs, "speed", 0.0)))
            px, py, pz = obs.position
            pos.append([float(px), float(py), float(pz)])
            actions.append([float(action.steering_angle), float(action.throttle)])

            # tick sync
            while obs.time == last_time:
                time.sleep(0.0025)
                obs = env.observe()

            if (step % 200) == 0 and step > 0:
                print(f"[debug] step={step} offtrack={offtrack_count} collisions={collision_count}")

            step += 1

    finally:
        wall = time.perf_counter() - t0
        delta = _settle_metrics(env, metrics_base, settle_seconds=1.5, probe_hz=10.0)
        offtrack_count = max(offtrack_count, int(delta["outOfTrackCount"]))
        collision_count = max(collision_count, int(delta["collisionCount"]))
        print(f"[episode] generalization: steps={step} wall={wall:.1f}s "
              f"offtrack_count={offtrack_count} collisions={collision_count}")

    # Keep schema identical: use perturbation_function="none", scale=0
    outcome = ScenarioOutcomeLite(
        frames=frames,
        pos=pos,
        xte=xte,
        speeds=speeds,
        actions=actions,
        pid_actions=[],
        scenario=ScenarioLite(perturbation_function="none", perturbation_scale=0, road=MAP_NAME),
        isSuccess=True,
        timeout=False,
        original_images=images,
        perturbed_images=None,
        offtrack_count=int(offtrack_count),
        collision_count=int(collision_count),
    )
    return outcome


def main() -> int:
    sim_app = _abs(getattr(paths, "SIM", getattr(paths, "SIM", "")))
    model_name = getattr(run, "MODEL_NAME", "dave2")

    if model_name == "dave2":
        ckpt = _abs(paths.DAVE2_CKPT) if getattr(paths, "DAVE2_CKPT", None) else None
    elif model_name == "dave2_gru":
        ckpt = _abs(paths.DAVE2_GRU_CKPT) if getattr(paths, "DAVE2_GRU_CKPT", None) else None
    else:
        raise ValueError(f"Unknown MODEL_NAME '{model_name}' in sims/udacity/configs/jungle/run.py")

    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit sims/udacity/configs/jungle/paths.py")
    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"{model_name.upper()}_CKPT not found: {ckpt}\nEdit sims/udacity/configs/jungle/paths.py")

    # Build adapter
    adapter = _build_adapter(model_name, image_size_hw=run.IMAGE_SIZE, ckpt_path=ckpt)

    # Logging
    ckpt_name = ckpt.stem if ckpt is not None else model_name
    _, run_dir = prepare_run_dir(
        map_name="jungle",
        test_type="generalization",
        model_name=model_name,
        tag=ckpt_name,
    )
    print(f"[drive:{MAP_NAME}] model={model_name} logs -> {run_dir}")

    git_info = {
        "thesis_sha": best_effort_git_sha(ROOT),
        "perturbation_drive_sha": best_effort_git_sha(PD_DIR),
    }
    logger = RunLogger(
        run_dir=run_dir,
        model=model_name,
        checkpoint=(str(ckpt) if ckpt else None),
        sim_name="udacity",
        git_info=git_info,
    )
    logger.snapshot_configs(
        sim_app=sim_app,
        ckpt=ckpt,
        cfg_paths=module_public_dict(paths),
        cfg_roads={"map": MAP_NAME},
        cfg_perturbations={},  # none for generalization
        cfg_run=module_public_dict(run),
        cfg_host_port={"host": HOST, "port": PORT},
    )
    logger.snapshot_env(pip_freeze())

    max_steps  = int(getattr(run, "STEPS", 2000))
    episodes   = int(getattr(run, "EPISODES", 1))
    save_images = bool(getattr(run, "SAVE_IMAGES", False))
    show_image  = bool(getattr(run, "SHOW_IMAGE", True))

    sim = UdacitySimulator(sim_exe_path=str(sim_app), host=HOST, port=PORT)
    env = UdacityGym(simulator=sim)
    sim.start()
    _force_start_episode(env,
                         track=MAP_NAME,
                         weather=getattr(run, "WEATHER", "sunny"),
                         daytime=getattr(run, "DAYTIME", "day"))
    preview = PDPreviewCallback(enabled=show_image)

    ep_idx = 0
    try:
        for ep in range(episodes):
            ep_idx += 1
            meta = {
                "road": MAP_NAME,
                "angles": None,
                "segs": None,
                "start": int(START_WAYPOINT),
                "perturbation": "none",
                "severity": 0,
                "image_size": {"h": run.IMAGE_SIZE[0], "w": run.IMAGE_SIZE[1]},
                "episodes": int(episodes),
                "ckpt_name": ckpt_name,
            }
            eid, ep_dir = logger.new_episode(ep_idx, meta)
            writer = ScenarioOutcomeWriter(str(ep_dir / "logs.json"), overwrite_logs=True)
            log_file = ep_dir / "pd_log.json"   # keep filename for downstream tools

            t0 = time.perf_counter()
            try:
                outcome = _run_episode(
                    env=env,
                    adapter=adapter,
                    preview=preview,
                    max_steps=max_steps,
                    save_images=save_images,
                )
                writer.write([outcome], images=save_images)
                with open(log_file, "w") as f:
                    f.write("{}\n")

                status = "ok" if outcome.isSuccess else "fail_offtrack"
                logger.complete_episode(eid, status=status, wall_time_s=(time.perf_counter() - t0))

            except Exception as e:
                logger.complete_episode(eid, status=f"error:{type(e).__name__}", wall_time_s=(time.perf_counter() - t0))
                raise

            # Soft reset + respawn for next episode
            try:
                adapter.reset()
            except Exception:
                pass
            _force_start_episode(env,
                                 track=MAP_NAME,
                                 weather=getattr(run, "WEATHER", "sunny"),
                                 daytime=getattr(run, "DAYTIME", "day"))
            _spawn_waypoint(env, START_WAYPOINT)

    except KeyboardInterrupt:
        print(f"\n[drive:{MAP_NAME}] Ctrl-C — stopping.")
    finally:
        try:
            preview.close()
        except Exception:
            pass
        try:
            sim.close()
            env.close()
        except Exception:
            pass
        print(f"[drive:{MAP_NAME}] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
