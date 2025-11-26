"""
Generalization evaluation of a model on the Udacity jungle map.

Arguments:
    --model MODEL_NAME
        Override experiment.default_model from eval/jungle/cfg_generalization.yaml.
        Examples: --model dave2, --model dave2_gru, --model vit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from external.udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from external.udacity_gym.agent_callback import PDPreviewCallback
from external.udacity_gym.logger import ScenarioOutcomeWriter, ScenarioOutcomeLite, ScenarioLite

from scripts import ROOT, abs_path
from scripts.udacity.adapters.dave2_adapter import Dave2Adapter
from scripts.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter
from scripts.udacity.logging.eval_runs import RunLogger, prepare_run_dir, best_effort_git_sha, pip_freeze


def load_cfg() -> dict:
    cfg_path = Path(__file__).with_name("cfg_generalization.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_adapter(model_name: str, model_cfg: dict, ckpts_dir: Path):
    ckpt_rel = model_cfg.get("checkpoint")
    ckpt = abs_path(ckpts_dir / ckpt_rel) if ckpt_rel else None

    image_size_hw = tuple(model_cfg.get("image_size_hw", [240, 320]))
    normalize = model_cfg.get("normalize", "imagenet")

    if model_name == "dave2":
        return (Dave2Adapter(weights=ckpt, image_size_hw=image_size_hw, device=None, normalize=normalize), ckpt)

    if model_name == "dave2_gru":
        return (Dave2GRUAdapter(weights=ckpt, image_size_hw=image_size_hw, device=None, normalize=normalize), ckpt)

    raise ValueError(f"Model '{model_name}' not defined under models in cfg_generalization.yaml")


def force_start_episode(env: UdacityGym, track: str, weather: str, daytime: str, timeout_s: float = 30.0) -> None:
    """
    Enter the map and wait until frames arrive (no Unity restart).
    """
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
            raise RuntimeError(f"Failed to enter map '{track}' before timeout.")

        time.sleep(0.2)


def await_scene_ready(env: UdacityGym, min_frames: int = 8, max_wait_s: float = 5.0) -> None:
    """
    Wait until Unity streams a few distinct frames.
    """
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


def metrics(env: UdacityGym) -> dict:
    try:
        m = env.simulator.get_episode_metrics() or {}
    except Exception:
        m = {}

    return {
        "outOfTrackCount": int(m.get("outOfTrackCount", 0)),
        "collisionCount": int(m.get("collisionCount", 0)),
    }


def settle_metrics(env: UdacityGym, prev: dict, settle_seconds: float = 1.5, probe_hz: float = 10.0) -> dict:
    """
    Let Unity flush its counters; return the per-episode deltas.
    """
    import time as _t

    delay = 1.0 / max(probe_hz, 1.0)
    deadline = _t.perf_counter() + max(settle_seconds, 0.0)
    last = metrics(env)

    while _t.perf_counter() < deadline:
        _t.sleep(delay)
        cur = metrics(env)
        if cur == last:
            break
        last = cur

    return {
        "outOfTrackCount": max(0, last["outOfTrackCount"] - prev["outOfTrackCount"]),
        "collisionCount": max(0, last["collisionCount"] - prev["collisionCount"]),
    }


def spawn_waypoint(env: UdacityGym, idx: int) -> None:
    """
    Request a spawn at a given waypoint index.
    """
    await_scene_ready(env, min_frames=8, max_wait_s=5.0)

    print(f"[eval:jungle:generalization][INFO] requesting waypoint {idx}")
    env.simulator.sim_executor.request_spawn_waypoint(int(idx))
    time.sleep(0.6)
    print(f"[eval:jungle:generalization][INFO] waypoint {idx} request dispatched")


def predict_to_action(adapter, image: np.ndarray) -> UdacityAction:
    """
    Run the adapter and convert its output into a UdacityAction.
    """
    try:
        out = getattr(adapter, "predict")(image)
    except AttributeError:
        out = adapter(image)

    if isinstance(out, UdacityAction):
        return out

    if isinstance(out, dict):
        steer = out.get("steer", out.get("steering", out.get("steering_angle", 0.0)))
        thr = out.get("throttle", out.get("accel", out.get("throttle_cmd", 0.0)))
        return UdacityAction(
            steering_angle=float(np.clip(float(steer), -1.0, 1.0)),
            throttle=float(np.clip(float(thr), 0.0, 1.0)),
        )

    arr = np.array(out).reshape(-1)
    return UdacityAction(
        steering_angle=float(np.clip(float(arr[0]), -1.0, 1.0)),
        throttle=float(np.clip(float(arr[1]), 0.0, 1.0)),
    )


def run_episode(env: UdacityGym, adapter, preview: PDPreviewCallback, max_steps: int, save_images: bool, track: str, weather: str, daytime: str, start_waypoint: int, image_size_hw: tuple[int, int]) -> ScenarioOutcomeLite:
    """
    Run a single generalization episode without perturbations.
    """
    force_start_episode(env, track=track, weather=weather, daytime=daytime)
    spawn_waypoint(env, start_waypoint)

    metrics_base = metrics(env)

    frames: list[int] = []
    xte: list[float] = []
    angle_diff: list[float] = []
    speeds: list[float] = []
    actions: list[list[float]] = []
    pos: list[list[float]] = []
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

            action = predict_to_action(adapter, img_np)
            preview(obs, display_image_np=img_np, action=action, perturbation="none")

            last_time = obs.time
            obs, reward, terminated, truncated, info = env.step(action)

            events = (info or {}).get("events") or []
            for e in events:
                key = (e.get("key") or e.get("type") or "").lower()
                if key == "out_of_track":
                    offtrack_count += 1
                elif key == "collision":
                    collision_count += 1

            if (info or {}).get("out_of_track") is True:
                offtrack_count += 1
            if (info or {}).get("collision") is True:
                collision_count += 1

            frames.append(step)
            xte.append(float(getattr(obs, "cte", 0.0)))
            angle_diff.append(float((obs.get_metrics() or {}).get("angleDiff", 0.0)))
            speeds.append(float(getattr(obs, "speed", 0.0)))
            px, py, pz = obs.position
            pos.append([float(px), float(py), float(pz)])
            actions.append([float(action.steering_angle), float(action.throttle)])

            while obs.time == last_time:
                time.sleep(0.0025)
                obs = env.observe()

            if step > 0 and (step % 200) == 0:
                print(f"[eval:jungle:generalization][DEBUG] step={step} offtrack={offtrack_count} collisions={collision_count}")

            step += 1

    finally:
        wall = time.perf_counter() - t0
        delta = settle_metrics(env, metrics_base, settle_seconds=1.5, probe_hz=10.0)
        offtrack_count = max(offtrack_count, int(delta["outOfTrackCount"]))
        collision_count = max(collision_count, int(delta["collisionCount"]))
        print(
            f"[eval:jungle:generalization][INFO] episode: steps={step} wall={wall:.1f}s "
            f"offtrack_count={offtrack_count} collisions={collision_count}"
        )

    return ScenarioOutcomeLite(
        frames=frames,
        pos=pos,
        xte=xte,
        speeds=speeds,
        actions=actions,
        pid_actions=[],
        scenario=ScenarioLite(
            perturbation_function="none",
            perturbation_scale=0,
            road=track,
        ),
        isSuccess=True,
        timeout=False,
        original_images=images,
        perturbed_images=None,
        offtrack_count=int(offtrack_count),
        collision_count=int(collision_count),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override experiment.default_model from eval/jungle/cfg_generalization.yaml. "
            "Examples: --model dave2, --model dave2_gru"
        ),
    )
    args = parser.parse_args()

    cfg = load_cfg()

    exp_cfg = cfg["experiment"]
    paths_cfg = cfg["paths"]
    sim_cfg = cfg["sim"]
    models_cfg = cfg["models"]
    run_cfg = cfg["run"]
    pert_cfg = cfg.get("perturbations", {})

    model_name = args.model or exp_cfg.get("default_model", "dave2")
    if model_name not in models_cfg:
        raise ValueError(f"Model '{model_name}' not defined under models in cfg_generalization.yaml")

    ckpts_dir = abs_path(paths_cfg["ckpts_dir"])
    runs_root = abs_path(paths_cfg["runs_dir"])
    data_dir = abs_path(paths_cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    adapter, ckpt = build_adapter(model_name, models_cfg[model_name], ckpts_dir)

    sim_app = abs_path(sim_cfg["binary"])
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit sim.binary in eval/jungle/cfg_generalization.yaml")

    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(
            f"{model_name.upper()} checkpoint not found: {ckpt}\n"
            "Edit models.*.checkpoint in eval/jungle/cfg_generalization.yaml"
        )

    ckpt_name = ckpt.stem if ckpt is not None else model_name
    map_name = exp_cfg.get("map", "jungle")

    _, run_dir = prepare_run_dir(
        map_name=map_name,
        test_type=exp_cfg.get("test_type", "generalization"),
        model_name=model_name,
        tag=ckpt_name,
    )
    print(f"[eval:jungle:generalization][INFO] model={model_name} logs -> {run_dir}")

    git_info = {
        "thesis_sha": best_effort_git_sha(ROOT),
        "perturbation_drive_sha": best_effort_git_sha(ROOT / "external" / "perturbation-drive"),
    }

    logger = RunLogger(
        run_dir=run_dir,
        model=model_name,
        checkpoint=str(ckpt) if ckpt else None,
        sim_name="udacity",
        git_info=git_info,
    )
    logger.snapshot_configs(
        sim_app=str(sim_app),
        ckpt=str(ckpt) if ckpt else None,
        cfg_paths=paths_cfg,
        cfg_roads={"map": map_name},
        cfg_perturbations=pert_cfg,
        cfg_run=run_cfg,
        cfg_host_port={"host": sim_cfg["host"], "port": sim_cfg["port"]},
    )
    logger.snapshot_env(pip_freeze())

    image_size_hw = tuple(run_cfg.get("image_size_hw", [240, 320]))
    max_steps = int(run_cfg.get("steps_per_episode", 2000))
    save_images = bool(run_cfg.get("save_images", False))
    show_image = bool(run_cfg.get("show_image", True))
    episodes = int(run_cfg.get("episodes", 1))
    start_waypoint = int(run_cfg.get("start_waypoint", 200))

    sim = UdacitySimulator(sim_exe_path=str(sim_app), host=sim_cfg["host"], port=int(sim_cfg["port"]))
    env = UdacityGym(simulator=sim)
    sim.start()

    force_start_episode(
        env,
        track=map_name,
        weather=sim_cfg.get("weather", "sunny"),
        daytime=sim_cfg.get("daytime", "day"),
    )
    preview = PDPreviewCallback(enabled=show_image)

    ep_idx = 0

    try:
        for _ in range(episodes):
            ep_idx += 1
            meta = {
                "road": map_name,
                "angles": None,
                "segs": None,
                "start": int(start_waypoint),
                "perturbation": "none",
                "severity": 0,
                "image_size": {"h": image_size_hw[0], "w": image_size_hw[1]},
                "episodes": int(episodes),
                "ckpt_name": ckpt_name,
            }
            eid, ep_dir = logger.new_episode(ep_idx, meta)

            writer = ScenarioOutcomeWriter(str(ep_dir / "log.json"), overwrite_logs=True)

            t0 = time.perf_counter()
            try:
                outcome = run_episode(
                    env=env,
                    adapter=adapter,
                    preview=preview,
                    max_steps=max_steps,
                    save_images=save_images,
                    track=map_name,
                    weather=sim_cfg.get("weather", "sunny"),
                    daytime=sim_cfg.get("daytime", "day"),
                    start_waypoint=start_waypoint,
                    image_size_hw=image_size_hw,
                )
                writer.write([outcome], images=save_images)

                status = "ok" if outcome.isSuccess else "fail_offtrack"
                logger.complete_episode(
                    eid,
                    status=status,
                    wall_time_s=(time.perf_counter() - t0),
                )

            except Exception as e:
                logger.complete_episode(
                    eid,
                    status=f"error:{type(e).__name__}",
                    wall_time_s=(time.perf_counter() - t0),
                )
                raise

            try:
                adapter.reset()
            except Exception:
                pass

            force_start_episode(
                env,
                track=map_name,
                weather=sim_cfg.get("weather", "sunny"),
                daytime=sim_cfg.get("daytime", "day"),
            )
            spawn_waypoint(env, start_waypoint)

    except KeyboardInterrupt:
        print("\n[eval:jungle:generalization][WARN] Ctrl-C — stopping.")
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
        print("[eval:jungle:generalization][INFO] done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
