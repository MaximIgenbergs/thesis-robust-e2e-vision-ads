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
import numpy as np

from external.udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from external.udacity_gym.agent_callback import PDPreviewCallback
from external.udacity_gym.logger import ScenarioOutcomeWriter, ScenarioOutcomeLite, ScenarioLite

from scripts.udacity.logging.eval_runs import RunLogger, prepare_run_dir, best_effort_git_sha, pip_freeze
from scripts.udacity.adapters.utils.build_adapter import build_adapter
from scripts import abs_path, load_cfg

from perturbationdrive import ImagePerturbation


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


def run_episode(env: UdacityGym, adapter, preview: PDPreviewCallback, controller: ImagePerturbation | None, pert_name: str, severity: int, max_steps: int, save_images: bool, track: str, weather: str, daytime: str, timeout_s: float) -> ScenarioOutcomeLite:
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
                print(f"[eval:jungle:robustness][INFO] episode {pert_name}@{severity} timeout after {timeout_s:.1f}s")
                break

            if obs is None or obs.input_image is None:
                time.sleep(0.005)
                obs = env.observe()
                continue

            img_np = np.asarray(obs.input_image, dtype=np.uint8).copy()
            if save_images:
                original_images.append(img_np)

            if controller is not None and pert_name:
                pert_np = controller.perturbation(img_np, pert_name, int(severity))
            else:
                pert_np = img_np # Baseline: no perturbation, use original image
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

        print(f"[eval:jungle:robustness][INFO] episode {pert_name}@{severity}: steps={step} wall={wall:.1f}s offtrack_count={offtrack_count} collisions={collision_count} {'(interrupted)' if stop_requested else ''}")

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
        help=("Override experiment.default_model from eval/udacity/jungle/cfg_robustness.yaml.\nExamples: --model dave2, --model dave2_gru, --model vit"),
    )
    args = parser.parse_args()

    cfg = load_cfg("eval/jungle/cfg_robustness.yaml")

    udacity_cfg = cfg["udacity"]
    models_cfg = cfg["models"]
    run_cfg = cfg["run"]
    logging_cfg = cfg["logging"]
    pert_cfg = cfg["perturbations"]
    baseline = bool(pert_cfg.get("baseline", False))


    model_defs = {k: v for k, v in models_cfg.items() if k != "default_model"}

    model_name = args.model or models_cfg.get("default_model", "dave2")
    if model_name not in model_defs:
        raise ValueError(f"Model '{model_name}' not defined under models in eval/jungle/cfg_robustness.yaml")

    adapter, ckpt = build_adapter(model_name, model_defs[model_name])

    sim_app = abs_path(udacity_cfg["binary"])
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit udacity.binary in eval/jungle/cfg_robustness.yaml")

    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"{model_name.upper()} checkpoint not found: {ckpt}\nEdit models.{model_name}.checkpoint in eval/jungle/cfg_robustness.yaml")

    runs_root = abs_path(logging_cfg["runs_dir"])

    ckpt_name = ckpt.stem if ckpt is not None else model_name
    _, run_dir = prepare_run_dir(model_name=model_name, runs_root=runs_root)
    print(f"[eval:jungle:robustness][INFO] model={model_name} logs -> {run_dir}")

    include_git = logging_cfg.get("include_git_sha", {})
    git_info = {}

    if include_git.get("thesis_repo", True):
        git_info["thesis_sha"] = best_effort_git_sha(abs_path(""))

    if include_git.get("perturbation_drive", True):
        git_info["perturbation_drive_sha"] = best_effort_git_sha(abs_path("external/perturbation-drive"))

    logger = RunLogger(
        run_dir=run_dir,
        model=model_name,
        checkpoint=str(ckpt) if ckpt else None,
        sim_name="udacity",
        git_info=git_info,
    )

    if logging_cfg.get("snapshot_configs", True):
        logger.snapshot_configs(
            sim_app=str(sim_app),
            ckpt=str(ckpt) if ckpt else None,
            cfg_logging=logging_cfg,
            cfg_udacity=udacity_cfg,
            cfg_models=models_cfg,
            cfg_perturbations=pert_cfg,
            cfg_run=run_cfg,
            cfg_host_port={"host": udacity_cfg["host"], "port": udacity_cfg["port"]},
        )

    if logging_cfg.get("snapshot_env", True):
        logger.snapshot_env(pip_freeze())

    chunks: list[list[str]] = pert_cfg.get("chunks")
    severities = list(pert_cfg.get("severities", [1, 2, 3, 4]))
    episodes = int(pert_cfg.get("episodes", 1))
    max_steps = int(run_cfg.get("steps_per_episode", 2000))
    save_images = bool(run_cfg.get("save_images", False))
    show_image = bool(run_cfg.get("show_image", True))
    image_size_hw = tuple(run_cfg.get("image_size_hw", [240, 320]))
    timeout_s = float(run_cfg.get("timeout_s", 300.0))

    sim = UdacitySimulator(str(sim_app), udacity_cfg["host"], int(udacity_cfg["port"]))
    env = UdacityGym(simulator=sim)
    sim.start()

    track = udacity_cfg.get("map", "jungle")
    weather = udacity_cfg.get("weather", "sunny")
    daytime = udacity_cfg.get("daytime", "day")

    force_start_episode(env, track=track, weather=weather, daytime=daytime)
    preview = PDPreviewCallback(enabled=show_image)

    ep_idx = 0

    # Baseline: one run on the jungle map without any perturbation
    if baseline:
        ep_idx += 1
        meta = {
            "road": track,
            "angles": None,
            "segs": None,
            "start": None,
            "perturbation": None,
            "severity": 0,
            "image_size": {"h": image_size_hw[0], "w": image_size_hw[1]},
            "episodes": 1,
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
                controller=None,  # no perturbation
                pert_name="",
                severity=0,
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

    try:
        for chunk in chunks:
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
