"""
Generalization evaluation of a model on the Udacity jungle map.

Arguments:
    --model MODEL_NAME
        Override experiment.default_model from eval/jungle/cfg_generalization.yaml.
        Examples: --model dave2, --model dave2_gru, --model vit
"""

from __future__ import annotations
import argparse
import contextlib
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from external.udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from external.udacity_gym.agent import PIDUdacityAgent_Angle
from external.udacity_gym.agent_callback import PDPreviewCallback
from external.udacity_gym.logger import ScenarioOutcomeWriter, ScenarioOutcomeLite, ScenarioLite

from scripts import abs_path, load_cfg
from scripts.udacity.adapters.utils.build_adapter import build_adapter
from scripts.udacity.logging.eval_runs import RunLogger, prepare_run_dir, best_effort_git_sha, pip_freeze


PID_TARGET_SPEED = 22.0


def compute_pid_action(pid_agent: PIDUdacityAgent_Angle, obs) -> UdacityAction:
    """
    Compute a reference PID action from the SAME pre-step observation.
    PID prints every step in the agent code -> silence stdout here.
    """
    try:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            a = pid_agent(obs)
    except Exception:
        return UdacityAction(steering_angle=0.0, throttle=0.0)

    if isinstance(a, UdacityAction):
        steer = float(np.clip(float(a.steering_angle), -1.0, 1.0))
        thr = float(np.clip(float(a.throttle), -1.0, 1.0))
        return UdacityAction(steering_angle=steer, throttle=thr)

    # fallback if agent ever returns array-like
    try:
        arr = np.array(a).reshape(-1)
        steer = float(np.clip(float(arr[0]), -1.0, 1.0))
        thr = float(np.clip(float(arr[1]), -1.0, 1.0))
        return UdacityAction(steering_angle=steer, throttle=thr)
    except Exception:
        return UdacityAction(steering_angle=0.0, throttle=0.0)


def force_start_episode(
    env: UdacityGym,
    track: str,
    weather: str,
    daytime: str,
    timeout_s: float = 30.0,
) -> None:
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


def spawn_waypoint(env: UdacityGym, idx: int) -> None:
    """
    Request a spawn at a given waypoint index (sector).
    """
    await_scene_ready(env, min_frames=8, max_wait_s=5.0)

    print(f"[eval:jungle:generalization][INFO] requesting waypoint {idx}")
    env.simulator.sim_executor.request_spawn_waypoint(int(idx))
    time.sleep(0.6)
    print(f"[eval:jungle:generalization][INFO] waypoint {idx} request dispatched")


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


def build_segments(run_cfg: dict, segments_cfg: list | None) -> List[Dict[str, Any]]:
    """
    Identical segment construction logic as robustness.
    """
    default_timeout_s = float(run_cfg.get("timeout_s", 300.0))

    segments: List[Dict[str, Any]] = []
    if segments_cfg:
        for seg in segments_cfg:
            segments.append(
                {
                    "id": seg.get("id", "seg"),
                    "start_waypoint": int(seg["start_waypoint"]),
                    "end_waypoint": int(seg["end_waypoint"]) if seg.get("end_waypoint") is not None else None,
                    "timeout_s": float(seg.get("timeout_s", default_timeout_s)),
                }
            )
    else:
        start_wp = int(run_cfg.get("start_waypoint", 200))
        segments.append(
            {
                "id": "full",
                "start_waypoint": start_wp,
                "end_waypoint": None,
                "timeout_s": float(default_timeout_s),
            }
        )

    return segments


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
            throttle=float(np.clip(float(thr), -1.0, 1.0)),
        )

    arr = np.array(out).reshape(-1)
    return UdacityAction(
        steering_angle=float(np.clip(float(arr[0]), -1.0, 1.0)),
        throttle=float(np.clip(float(arr[1]), -1.0, 1.0)),
    )


def get_sector(obs) -> int | None:
    """
    Read the current sector (waypoint index) from obs.get_metrics().
    """
    try:
        m = obs.get_metrics() or {}
    except Exception:
        return None

    sec = m.get("sector")
    if sec is None:
        return None
    try:
        return int(sec)
    except (TypeError, ValueError):
        return None


def run_segment(
    env: UdacityGym,
    adapter,
    preview: PDPreviewCallback,
    save_images: bool,
    track: str,
    weather: str,
    daytime: str,
    timeout_s: float,
    start_waypoint: int | None = None,
    end_waypoint: int | None = None,
) -> Tuple[ScenarioOutcomeLite, List[Dict[str, Any]]]:
    """
    Run one segment (no perturbations).

    Termination:
      - segment timeout_s reached, or
      - (if provided) reached end_waypoint (wrap-aware crossing), or
      - env signals terminated or truncated, or
      - KeyboardInterrupt.
    """
    force_start_episode(env, track=track, weather=weather, daytime=daytime)

    if start_waypoint is not None:
        spawn_waypoint(env, start_waypoint)

    metrics_base = metrics(env)

    # Reference PID (fresh per segment, no carryover)
    pid_agent = PIDUdacityAgent_Angle(
        target_speed=PID_TARGET_SPEED,
        track="jungle",
        before_action_callbacks=[],
        after_action_callbacks=[],
    )

    frames: list[int] = []
    xte: list[float] = []
    angle_diff: list[float] = []
    speeds: list[float] = []
    actions: list[list[float]] = []
    pid_actions: list[list[float]] = []
    pos: list[list[float]] = []
    images = [] if save_images else None

    events_log: List[Dict[str, Any]] = []

    offtrack_count = 0
    collision_count = 0
    stop_requested = False
    timed_out = False
    reached_end = False

    step = 0
    t0 = time.perf_counter()
    obs = env.observe()
    prev_sector: int | None = None  # for wrap-aware end detection

    # state for event de-duplication (same as robustness)
    seen_events: set[tuple[str, str]] = set()
    last_flag_out = False
    last_flag_collision = False

    try:
        while True:
            if timeout_s is not None and (time.perf_counter() - t0) > timeout_s:
                timed_out = True
                print(f"[eval:jungle:generalization][INFO] segment timeout after {timeout_s:.1f}s")
                break

            if obs is None or obs.input_image is None:
                time.sleep(0.005)
                obs = env.observe()
                continue

            # PID reference on SAME pre-step obs
            pid_a = compute_pid_action(pid_agent, obs)

            img_np = np.asarray(obs.input_image, dtype=np.uint8).copy()
            if save_images:
                images.append(img_np)

            action = predict_to_action(adapter, img_np)
            preview(obs, display_image_np=img_np, action=action, perturbation="none")

            last_time = obs.time
            obs, reward, terminated, truncated, info = env.step(action)

            # Log discrete events with timestamps for offline metrics (deduplicated)
            events = (info or {}).get("events") or []
            for e in events:
                key = (e.get("key") or e.get("type") or "").lower()
                if key in ("out_of_track", "collision"):
                    ts = str(e.get("timestamp", ""))
                    ident = (ts, key)
                    if ident in seen_events:
                        continue
                    seen_events.add(ident)

                    events_log.append(
                        {"step": int(step), "sim_time": float(last_time), "event": key, "raw": e}
                    )
                    if key == "out_of_track":
                        offtrack_count += 1
                    elif key == "collision":
                        collision_count += 1

            # Compatibility with possible boolean flags in info (only count rising edge)
            flag_out = (info or {}).get("out_of_track") is True
            if flag_out and not last_flag_out:
                events_log.append(
                    {"step": int(step), "sim_time": float(last_time), "event": "out_of_track", "raw": {"source": "flag"}}
                )
                offtrack_count += 1
            last_flag_out = flag_out

            flag_collision = (info or {}).get("collision") is True
            if flag_collision and not last_flag_collision:
                events_log.append(
                    {"step": int(step), "sim_time": float(last_time), "event": "collision", "raw": {"source": "flag"}}
                )
                collision_count += 1
            last_flag_collision = flag_collision

            frames.append(step)
            xte.append(float(getattr(obs, "cte", 0.0)))
            angle_diff.append(float(getattr(obs, "angle_diff", 0.0)))
            speeds.append(float(getattr(obs, "speed", 0.0)))
            px, py, pz = obs.position
            pos.append([float(px), float(py), float(pz)])
            actions.append([float(action.steering_angle), float(action.throttle)])
            pid_actions.append([float(pid_a.steering_angle), float(pid_a.throttle)])

            # End-waypoint termination (wrap-aware, same logic as robustness)
            hit_end = False
            sector_for_log: int | None = None
            if end_waypoint is not None:
                sector = get_sector(obs)
                if sector is not None:
                    if prev_sector is None:
                        if sector >= end_waypoint:
                            hit_end = True
                            sector_for_log = sector
                    else:
                        if prev_sector < end_waypoint <= sector:
                            hit_end = True
                            sector_for_log = sector
                        elif prev_sector > sector:
                            hit_end = True
                            sector_for_log = sector
                    prev_sector = sector

            if hit_end:
                print(
                    f"[eval:jungle:generalization][INFO] segment reached end_waypoint={end_waypoint} (sector={sector_for_log})"
                )
                reached_end = True
                step += 1
                break

            while obs.time == last_time:
                time.sleep(0.0025)
                obs = env.observe()

            step += 1

            if terminated or truncated:
                print(
                    f"[eval:jungle:generalization][INFO] episode terminated: terminated={terminated} truncated={truncated}"
                )
                break

    except KeyboardInterrupt:
        stop_requested = True
        print(
            "\n[eval:jungle:generalization][WARN] segment interrupted by user. Attempting to finish writing logs..."
        )
    finally:
        wall = time.perf_counter() - t0

        if reached_end and not timed_out and not stop_requested:
            delta = settle_metrics(env, metrics_base, settle_seconds=0.0, probe_hz=10.0)
        else:
            delta = settle_metrics(env, metrics_base, settle_seconds=1.5, probe_hz=10.0)

        offtrack_count = max(offtrack_count, int(delta["outOfTrackCount"]))
        collision_count = max(collision_count, int(delta["collisionCount"]))

        print(
            f"[eval:jungle:generalization][INFO] segment: steps={step} wall={wall:.1f}s "
            f"offtrack_count={offtrack_count} collisions={collision_count} "
            f"{'(interrupted)' if stop_requested else ''}"
        )

    is_success = reached_end and not timed_out and not stop_requested

    outcome = ScenarioOutcomeLite(
        frames=frames,
        pos=pos,
        xte=xte,
        angle_diff=angle_diff,
        speeds=speeds,
        actions=actions,
        pid_actions=pid_actions,
        scenario=ScenarioLite(
            perturbation_function="none",
            perturbation_scale=0,
            road=track,
        ),
        isSuccess=bool(is_success),
        timeout=bool(stop_requested),
        original_images=images,
        perturbed_images=None,
        offtrack_count=int(offtrack_count),
        collision_count=int(collision_count),
    )

    return outcome, events_log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override experiment.default_model from eval/jungle/cfg_generalization.yaml.\n"
            "Examples: --model dave2, --model dave2_gru, --model vit"
        ),
    )
    args = parser.parse_args()

    cfg = load_cfg("eval/jungle/cfg_generalization.yaml")

    udacity_cfg = cfg["udacity"]
    models_cfg = cfg["models"]
    run_cfg = cfg["run"]
    segments_cfg = cfg.get("segments")
    logging_cfg = cfg["logging"]

    model_defs = {k: v for k, v in models_cfg.items() if k != "default_model"}

    model_name = args.model or models_cfg.get("default_model", "dave2")
    if model_name not in model_defs:
        raise ValueError(f"Model '{model_name}' not defined under models in cfg_generalization.yaml")

    adapter, ckpt = build_adapter(model_name, model_defs[model_name])

    sim_app = abs_path(udacity_cfg["binary"])
    if not sim_app.exists():
        raise FileNotFoundError(
            f"SIM not found: {sim_app}\nEdit udacity.binary in eval/jungle/cfg_generalization.yaml"
        )

    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(
            f"{model_name.upper()} checkpoint not found: {ckpt}\n"
            f"Edit models.{model_name}.checkpoint in eval/jungle/cfg_generalization.yaml"
        )

    runs_root = abs_path(logging_cfg["runs_dir"])

    ckpt_name = ckpt.stem if ckpt is not None else model_name
    map_name = udacity_cfg.get("map", "jungle")
    _, run_dir = prepare_run_dir(model_name=model_name, runs_root=runs_root)
    print(f"[eval:jungle:generalization][INFO] model={model_name} logs -> {run_dir}")

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
            cfg_run=run_cfg,
            cfg_segments=segments_cfg,
            cfg_host_port={"host": udacity_cfg["host"], "port": udacity_cfg["port"]},
        )

    if logging_cfg.get("snapshot_env", True):
        logger.snapshot_env(pip_freeze())

    image_size_hw = tuple(run_cfg.get("image_size_hw", [240, 320]))
    save_images = bool(run_cfg.get("save_images", False))
    show_image = bool(run_cfg.get("show_image", True))
    episodes = int(run_cfg.get("episodes", 1))

    segments = build_segments(run_cfg, segments_cfg)

    sim = UdacitySimulator(str(sim_app), udacity_cfg["host"], int(udacity_cfg["port"]))
    env = UdacityGym(simulator=sim)
    sim.start()

    track = map_name
    weather = udacity_cfg.get("weather", "sunny")
    daytime = udacity_cfg.get("daytime", "day")

    force_start_episode(env, track=track, weather=weather, daytime=daytime)
    preview = PDPreviewCallback(enabled=show_image)

    ep_idx = 0

    try:
        for _ in range(episodes):
            ep_idx += 1

            meta = {
                "road": track,
                "angles": None,
                "segs": [s["id"] for s in segments],
                "start": None,
                "segment_id": None,
                "segment_end_waypoint": None,
                "segment_timeout_s": float(run_cfg.get("timeout_s", 300.0)),
                "perturbation": "none",
                "severity": 0,
                "image_size": {"h": image_size_hw[0], "w": image_size_hw[1]},
                "episodes": int(episodes),
                "ckpt_name": ckpt_name,
                "pid": {"target_speed": float(PID_TARGET_SPEED), "track": "jungle"},
            }
            eid, ep_dir = logger.new_episode(ep_idx, meta)

            writer = ScenarioOutcomeWriter(str(ep_dir / "log.json"), overwrite_logs=True)

            all_outcomes: List[ScenarioOutcomeLite] = []
            all_events: List[Dict[str, Any]] = []

            t0 = time.perf_counter()
            try:
                for seg in segments:
                    seg_id = seg["id"]
                    start_wp = seg["start_waypoint"]
                    seg_end_wp = seg["end_waypoint"]
                    seg_timeout_s = seg["timeout_s"]

                    outcome, events_log = run_segment(
                        env=env,
                        adapter=adapter,
                        preview=preview,
                        save_images=save_images,
                        track=track,
                        weather=weather,
                        daytime=daytime,
                        timeout_s=seg_timeout_s,
                        start_waypoint=start_wp,
                        end_waypoint=seg_end_wp,
                    )

                    for e in events_log:
                        e.setdefault("segment_id", seg_id)

                    all_outcomes.append(outcome)
                    all_events.extend(events_log)

                    try:
                        adapter.reset()
                    except Exception:
                        pass

                    force_start_episode(env, track=track, weather=weather, daytime=daytime)

                writer.write(all_outcomes, images=save_images)

                (ep_dir / "events.json").write_text(
                    json.dumps(all_events, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

                if any(o.timeout for o in all_outcomes):
                    status = "interrupted"
                elif all(o.isSuccess for o in all_outcomes):
                    status = "ok"
                else:
                    status = "fail_offtrack"

                logger.complete_episode(eid, status=status, wall_time_s=(time.perf_counter() - t0))

            except Exception as e:
                logger.complete_episode(
                    eid,
                    status=f"error:{type(e).__name__}",
                    wall_time_s=(time.perf_counter() - t0),
                )
                raise

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
