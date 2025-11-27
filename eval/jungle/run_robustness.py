"""
Robustness evaluation of a model on the Udacity jungle map using image perturbations from PerturbationDrive.

Arguments:
    --model MODEL_NAME
        Override experiment.default_model from eval/jungle/cfg_robustness.yaml.
        Examples: --model dave2, --model dave2_gru, --model vit
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import yaml

import sys
import multiprocessing as mp

if sys.platform == "darwin":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

from external.udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from external.udacity_gym.agent_callback import PDPreviewCallback
from external.udacity_gym.logger import ScenarioOutcomeWriter, ScenarioOutcomeLite, ScenarioLite

from scripts.udacity.logging.eval_runs import RunLogger, prepare_run_dir, best_effort_git_sha, pip_freeze
from scripts.udacity.adapters.dave2_adapter import Dave2Adapter
from scripts.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter
from scripts import abs_path

from perturbationdrive import ImagePerturbation


def load_cfg() -> dict:
    cfg_path = Path(__file__).with_name("cfg_robustness.yaml")
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
        seq_len = int(model_cfg.get("sequence_length", 3))
        return (Dave2GRUAdapter(weights=ckpt, image_size_hw=image_size_hw, seq_len=seq_len, device=None, normalize=normalize), ckpt)

    raise ValueError(f"Unknown model '{model_name}' in eval/udacity/jungle/cfg_robustness.yaml")


def force_start_episode(env: UdacityGym, track: str = "jungle", weather: str = "sunny", daytime: str = "day", timeout_s: float = 30.0) -> None:
    """
    Robustly enter the map without restarting the Unity process.
    Keeps re-sending 'start_episode' until frames arrive.
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


def run_episode(env: UdacityGym, adapter, preview: PDPreviewCallback, controller: ImagePerturbation, pert_name: str, severity: int, max_steps: int, save_images: bool, track: str, weather: str, daytime: str, timeout_s: float) -> ScenarioOutcomeLite:
    """
    Run a single episode with one (perturbation, severity) pair.
    """
    force_start_episode(env, track=track, weather=weather, daytime=daytime)

    metrics_base = metrics(env)

    frames: list[int] = []
    xte: list[float] = []
    angle_diff: list[float] = []
    speeds: list[float] = []
    actions: list[list[float]] = []
    pos: list[list[float]] = []

    original_images = [] if save_images else None
    perturbed_images = [] if save_images else None

    offtrack_count = 0
    collision_count = 0
    stop_requested = False

    step = 0
    t0 = time.perf_counter()
    obs = env.observe()
    timed_out = False

    try:
        while step < max_steps:
            if timeout_s is not None and (time.perf_counter() - t0) > timeout_s:
                timed_out = True
                print(
                    f"[eval:jungle:robustness][INFO] episode {pert_name}@{severity} "
                    f"timeout after {timeout_s:.1f}s"
                )
                break

            if obs is None or obs.input_image is None:
                time.sleep(0.005)
                obs = env.observe()
                continue

            img_np = np.asarray(obs.input_image, dtype=np.uint8).copy()
            if save_images:
                original_images.append(img_np)

            pert_np = controller.perturbation(img_np, pert_name, int(severity))
            if save_images:
                perturbed_images.append(pert_np)

            try:
                out = getattr(adapter, "predict")(pert_np)
            except AttributeError:
                out = adapter(pert_np)

            if isinstance(out, UdacityAction):
                action = out
            elif isinstance(out, dict):
                steer = out.get("steer", out.get("steering", out.get("steering_angle", 0.0)))
                thr = out.get("throttle", out.get("accel", out.get("throttle_cmd", 0.0)))
                action = UdacityAction(
                    steering_angle=float(np.clip(float(steer), -1.0, 1.0)),
                    throttle=float(np.clip(float(thr), -1.0, 1.0)),
                )
            else:
                arr = np.array(out).reshape(-1)
                action = UdacityAction(
                    steering_angle=float(np.clip(float(arr[0]), -1.0, 1.0)),
                    throttle=float(np.clip(float(arr[1]), -1.0, 1.0)),
                )

            preview(obs, display_image_np=pert_np, action=action, perturbation=pert_name)

            last_time = obs.time
            obs, reward, terminated, truncated, info = env.step(action)

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

            step += 1

    except KeyboardInterrupt:
        stop_requested = True
        print(f"\n[eval:jungle:robustness][WARN] episode {pert_name}@{severity} interrupted by user."
              f"Attempting to finish writing logs...")
    finally:
        wall = time.perf_counter() - t0
        delta = settle_metrics(env, metrics_base, settle_seconds=1.5, probe_hz=10.0)
        offtrack_count = int(delta["outOfTrackCount"])
        collision_count = int(delta["collisionCount"])

        print(
            f"[eval:jungle:robustness][INFO] episode {pert_name}@{severity}: "
            f"steps={step} wall={wall:.1f}s "
            f"offtrack_count={offtrack_count} collisions={collision_count} "
            f"{'(interrupted)' if stop_requested else ''}"
        )

    is_success = not timed_out and not stop_requested

    return ScenarioOutcomeLite(
        frames=frames,
        pos=pos,
        xte=xte,
        angle_diff=angle_diff,
        speeds=speeds,
        actions=actions,
        pid_actions=[],
        scenario=ScenarioLite(
            perturbation_function=pert_name,
            perturbation_scale=int(severity),
            road=track,
        ),
        isSuccess=bool(is_success),
        timeout=bool(stop_requested),
        original_images=original_images,
        perturbed_images=perturbed_images,
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
            "Override experiment.default_model from eval/udacity/jungle/cfg_robustness.yaml. "
            "Examples: --model dave2, --model dave2_gru, --model vit"
        ),
    )
    args = parser.parse_args()

    cfg = load_cfg()

    exp_cfg = cfg["experiment"]
    paths_cfg = cfg["paths"]
    sim_cfg = cfg["sim"]
    models_cfg = cfg["models"]
    run_cfg = cfg["run"]
    pert_cfg = cfg["perturbations"]

    model_name = args.model or exp_cfg.get("default_model", "dave2")
    if model_name not in models_cfg:
        raise ValueError(f"Model '{model_name}' not defined under models in eval/jungle/cfg_robustness.yaml")

    ckpts_dir = abs_path(paths_cfg["ckpts_dir"])
    runs_root = abs_path(paths_cfg["runs_dir"])
    data_dir = abs_path(paths_cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    adapter, ckpt = build_adapter(model_name, models_cfg[model_name], ckpts_dir)

    sim_app = abs_path(sim_cfg["binary"])
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit sim.binary in eval/jungle/cfg_robustness.yaml")

    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"{model_name.upper()} checkpoint not found: {ckpt}\nEdit models.*.checkpoint in eval/jungle/cfg_robustness.yaml"
        )

    ckpt_name = ckpt.stem if ckpt is not None else model_name
    _, run_dir = prepare_run_dir(
        map_name=exp_cfg.get("map", "jungle"),
        test_type=exp_cfg.get("test_type", "robustness"),
        model_name=model_name,
    )
    print(f"[eval:jungle:robustness][INFO] model={model_name} logs -> {run_dir}")

    git_info = {
        "thesis_sha": best_effort_git_sha(abs_path("")),
        "perturbation_drive_sha": best_effort_git_sha(abs_path("external/perturbation-drive")),
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
        cfg_roads={"map": exp_cfg.get("map", "jungle")},
        cfg_perturbations=pert_cfg,
        cfg_run=run_cfg,
        cfg_host_port={"host": sim_cfg["host"], "port": sim_cfg["port"]},
    )
    logger.snapshot_env(pip_freeze())

    mode = pert_cfg.get("mode", "chunks" if "chunks" in pert_cfg else "list")
    if mode == "single":
        single = pert_cfg["single"]
        perturbations: list[list[str]] = [[single]]
    elif mode == "list":
        perturbations = [pert_cfg["list"]]
    else:  # "chunks"
        perturbations = pert_cfg.get("chunks") or [pert_cfg["list"]]

    severities = list(pert_cfg.get("severities", [1, 2, 3, 4]))
    episodes = int(pert_cfg.get("episodes", 1))
    max_steps = int(run_cfg.get("steps_per_episode", 2000))
    save_images = bool(run_cfg.get("save_images", False))
    show_image = bool(run_cfg.get("show_image", True))
    image_size_hw = tuple(run_cfg.get("image_size_hw", [240, 320]))
    timeout_s = float(run_cfg.get("timeout_s", 300.0))

    sim = UdacitySimulator(str(sim_app), sim_cfg["host"], int(sim_cfg["port"]))
    env = UdacityGym(simulator=sim)
    sim.start()

    track = exp_cfg.get("map", "jungle")
    weather = sim_cfg.get("weather", "sunny")
    daytime = sim_cfg.get("daytime", "day")

    force_start_episode(env, track=track, weather=weather, daytime=daytime)
    preview = PDPreviewCallback(enabled=show_image)

    ep_idx = 0

    try:
        for chunk in perturbations:
            controller = ImagePerturbation(funcs=list(chunk), image_size=image_size_hw)

            force_start_episode(env, track=track, weather=weather, daytime=daytime)

            for pert in chunk:
                for sev in severities:
                    for _ in range(episodes):
                        ep_idx += 1
                        meta = {
                            "road": track,
                            "angles": None,
                            "segs": None,
                            "start": None,
                            "perturbation": pert,
                            "severity": int(sev),
                            "image_size": {"h": image_size_hw[0], "w": image_size_hw[1]},
                            "episodes": int(episodes),
                            "ckpt_name": ckpt_name,
                        }
                        eid, ep_dir = logger.new_episode(ep_idx, meta)

                        writer = ScenarioOutcomeWriter(
                            str(ep_dir / "log.json"),
                            overwrite_logs=True,
                        )

                        t0 = time.perf_counter()
                        try:
                            outcome = run_episode(
                                env=env,
                                adapter=adapter,
                                preview=preview,
                                controller=controller,
                                pert_name=pert,
                                severity=int(sev),
                                max_steps=max_steps,
                                save_images=save_images,
                                track=track,
                                weather=weather,
                                daytime=daytime,
                                timeout_s=timeout_s,
                            )
                            writer.write([outcome], images=save_images)

                            if outcome.timeout:
                                status = "interrupted"
                            elif outcome.isSuccess:
                                status = "ok"
                            else:
                                status = "fail_offtrack"

                            logger.complete_episode(
                                eid,
                                status=status,
                                wall_time_s=(time.perf_counter() - t0),
                            )

                            if outcome.timeout:
                                raise KeyboardInterrupt

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

                        force_start_episode(env, track=track, weather=weather, daytime=daytime)

    except KeyboardInterrupt:
        print("\n[eval:jungle:robustness][WARN] Ctrl-C — stopping.")
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

        print("[eval:jungle:robustness][INFO] done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
