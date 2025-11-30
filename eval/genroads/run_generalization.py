"""
Generalization evaluation on Udacity genroads using control-space scenarios.

Perturbs steering/throttle over waypoint intervals and logs the resulting
behaviour per episode.

Arguments:
    --model MODEL_NAME
        Override experiment.default_model from cfg_generalization.yaml.
        Examples: --model dave2, --model dave2_gru
"""

from __future__ import annotations

import argparse
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gym
import numpy as np
import yaml

from scripts import abs_path
from scripts.udacity.adapters.dave2_adapter import Dave2Adapter
from scripts.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter
from scripts.udacity.logging.eval_runs import RunLogger, prepare_run_dir, best_effort_git_sha, pip_freeze
from scripts.udacity.maps.genroads.configs import roads

from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from perturbationdrive import ImageCallBack


def load_cfg() -> Dict[str, Any]:
    cfg_path = Path(__file__).with_name("cfg_generalization.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_adapter(model_name: str, model_cfg: Dict[str, Any], ckpts_dir: Path):
    ckpt_rel = model_cfg.get("checkpoint")
    ckpt = abs_path(ckpts_dir / ckpt_rel) if ckpt_rel else None

    image_size_hw = tuple(model_cfg.get("image_size_hw", [240, 320]))
    normalize = model_cfg.get("normalize", "imagenet")
    seq_len = int(model_cfg.get("sequence_length", 3))

    if model_name == "dave2":
        return Dave2Adapter(weights=ckpt, image_size_hw=image_size_hw, device=None, normalize=normalize), ckpt
    if model_name == "dave2_gru":
        return Dave2GRUAdapter(weights=ckpt, image_size_hw=image_size_hw, seq_len=seq_len, device=None, normalize=normalize), ckpt

    raise ValueError(f"Unknown model '{model_name}' in eval/genroads/cfg_generalization.yaml")


def nearest_wp_index(x: float, y: float, waypoints: Sequence[Tuple[float, float]]) -> int:
    best_i = 0
    best_d2 = float("inf")
    for i, (wx, wy) in enumerate(waypoints):
        dx = wx - x
        dy = wy - y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_i = i
            best_d2 = d2
    return best_i


class Phase(str, Enum):
    BEFORE = "before_setup"
    SETUP = "setup"
    REACTION = "reaction"


def scenario_bounds(scenario: Dict[str, Any]) -> tuple[int, int]:
    """Global [start, end] waypoint index range covered by this scenario."""
    perts = scenario.get("perturbations", [])
    if not perts:
        return 0, -1
    starts = [int(p["start_wp"]) for p in perts]
    ends = [int(p["end_wp"]) for p in perts]
    return min(starts), max(ends)


def classify_phase(wp_idx: int, scenario: Dict[str, Any]) -> Phase:
    start_wp, end_wp = scenario_bounds(scenario)
    if wp_idx < start_wp:
        return Phase.BEFORE
    if wp_idx <= end_wp:
        return Phase.SETUP
    return Phase.REACTION


def apply_scenario(scenario: Dict[str, Any], wp_idx: int, base_action: Tuple[float, float]) -> Tuple[Tuple[float, float], Phase, List[str], List[Dict[str, Any]]]:
    steer, throttle = base_action
    active_controls: List[str] = []
    active_perturbations: List[Dict[str, Any]] = []

    for p in scenario.get("perturbations", []):
        start_wp = int(p["start_wp"])
        end_wp = int(p["end_wp"])
        if not (start_wp <= wp_idx <= end_wp):
            continue

        ctrl = p["control"]
        mode = p["mode"]
        factor = float(p["factor"])

        active_controls.append(ctrl)
        active_perturbations.append(p)

        if ctrl == "steering":
            if mode == "mul":
                steer *= factor
            elif mode == "add":
                steer += factor
            elif mode == "set":
                steer = factor

        elif ctrl == "throttle":
            if mode == "mul":
                throttle *= factor
            elif mode == "add":
                throttle += factor
            elif mode == "set":
                throttle = factor

    steer = max(-1.0, min(1.0, steer))
    throttle = max(-1.0, min(1.0, throttle))

    phase = classify_phase(wp_idx, scenario)
    return (steer, throttle), phase, active_controls, active_perturbations


def _extract_pid_state(info: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for key in ("cte", "cte_pid", "angle", "speed"):
        if key in info:
            state[key] = info[key]
    return state


def format_active_perturbations(perts: List[Dict[str, Any]]) -> str:
    if not perts:
        return "none"

    parts: List[str] = []
    for p in perts:
        ctrl = p["control"]
        mode = p["mode"]
        factor = float(p["factor"])

        if ctrl == "steering":
            ctrl_label = "S"
        else:
            ctrl_label = "T"

        if mode == "mul":
            op = f"*{factor:.2f}"
        elif mode == "add":
            op = f"{factor:+.2f}"
        else:
            op = f"={factor:.2f}"

        parts.append(f"{ctrl_label}{op}")

    return " ".join(parts)


def run_scenario_episode(sim: UdacitySimulator, controller, track_string: str, waypoints: List[Tuple[float, float]], scenario: Dict[str, Any], log_path: Path, max_steps: int | None = None, enable_preview: bool = False) -> Tuple[str, float]:
    if sim.client is None:
        raise RuntimeError("UdacitySimulator.client is None. Did you call sim.connect()?")

    env = sim.client
    history: List[dict] = []
    monitor: ImageCallBack | None = None

    print(f"[eval:genroads:generalization][INFO] Applying scenario: {json.dumps(scenario, indent=2)}")

    try:
        if enable_preview:
            monitor = ImageCallBack()
            monitor.display_waiting_screen()

        obs = env.reset(skip_generation=False, track_string=track_string)
        obs, done, info = env.observe()

        start_time = time.perf_counter()
        step_idx = 0

        while not done:
            model_actions = controller.action(obs)
            model_actions = np.asarray(model_actions, dtype=np.float32)
            if model_actions.ndim == 1:
                model_actions = model_actions.reshape(1, -1)

            model_steer = float(model_actions[0][0])
            model_throttle = float(model_actions[0][1])

            pos = info.get("pos")
            x = float(pos[0])
            y = float(pos[1])
            wp_idx = nearest_wp_index(x, y, waypoints)

            print(f"[eval:genroads:generalization][INFO] step={step_idx:04d} wp_idx={wp_idx:4d}  x={x:7.2f}  y={y:7.2f}", flush=True)

            (act_steer, act_throttle), phase, active_controls, active_perts = apply_scenario(scenario, wp_idx, (model_steer, model_throttle))

            actual_actions = np.array([[act_steer, act_throttle]], dtype=np.float32)
            if isinstance(env.action_space, gym.spaces.Box):
                actual_actions = np.clip(actual_actions, env.action_space.low, env.action_space.high)

            pert_summary = format_active_perturbations(active_perts)

            if monitor is not None:
                phase_label = phase.value
                scenario_label = (
                    f"{scenario['name']} (level {scenario['level']}):\n"
                    f"Road: {scenario['road']}, Waypoint: {wp_idx}\n"
                    f"Phase: {phase_label.upper()} --> Perturbations: {pert_summary}\n"
                    f"Comment: {scenario.get('comment', '')}"
                )
                monitor.display_img(obs, f"{actual_actions[0][0]: .3f}", f"{actual_actions[0][1]: .3f}", scenario_label)

            obs_next, done, info_next = env.step(actual_actions)

            pid_state = _extract_pid_state(info_next)

            history.append(
                {
                    "step": int(step_idx),
                    "wp_idx": int(wp_idx),
                    "phase": phase.value,
                    "active_controls": active_controls,
                    "model_steer": float(model_steer),
                    "model_throttle": float(model_throttle),
                    "actual_steer": float(act_steer),
                    "actual_throttle": float(act_throttle),
                    "delta_steer": float(act_steer - model_steer),
                    "delta_throttle": float(act_throttle - model_throttle),
                    "pid_state": pid_state,
                    "pert_summary": pert_summary,
                    "reward": None,
                    "info": info_next,
                }
            )

            obs = obs_next
            info = info_next
            step_idx += 1

            if max_steps is not None and step_idx >= max_steps:
                break

        wall = time.perf_counter() - start_time
        with log_path.open("w", encoding="utf-8") as f:
            json.dump({"steps": history}, f, indent=2)
        return "ok", wall

    except Exception as e:
        wall = time.perf_counter() - start_time if "start_time" in locals() else 0.0
        with log_path.open("w", encoding="utf-8") as f:
            json.dump({"steps": history, "error": str(e)}, f, indent=2)
        return f"error:{type(e).__name__}", wall

    finally:
        if monitor is not None:
            try:
                monitor.display_disconnect_screen()
                monitor.destroy()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override experiment.default_model from eval/genroads/cfg_generalization.yaml. "
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

    model_name = args.model or exp_cfg.get("default_model", "dave2")
    if model_name not in models_cfg:
        raise ValueError(f"Model '{model_name}' not defined under models in eval/genroads/cfg_generalization.yaml")

    ckpts_dir = abs_path(paths_cfg["ckpts_dir"])
    runs_root = abs_path(paths_cfg["runs_dir"])
    runs_root.mkdir(parents=True, exist_ok=True)

    adapter, ckpt = build_adapter(model_name, models_cfg[model_name], ckpts_dir)

    sim_app = abs_path(sim_cfg["binary"])
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit sim.binary in eval/genroads/cfg_generalization.yaml")

    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"{model_name.upper()} checkpoint not found: {ckpt}\nEdit models.*.checkpoint in eval/genroads/cfg_generalization.yaml")

    scenarios_path = abs_path(paths_cfg["scenarios_path"])
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenario config not found: {scenarios_path}\nSet run.scenarios_path in eval/genroads/cfg_generalization.yaml")
    with scenarios_path.open("r", encoding="utf-8") as f:
        scenarios_by_road = yaml.safe_load(f) or {}

    ckpt_name = ckpt.stem if ckpt is not None else model_name
    _, run_dir = prepare_run_dir(exp_cfg.get("map", "genroads"), exp_cfg.get("test_type", "generalization"), model_name)
    print(f"[eval:genroads:generalization][INFO] model={model_name} logs -> {run_dir}")

    git_info = {
        "thesis_sha": best_effort_git_sha(abs_path("")),
        "perturbation_drive_sha": best_effort_git_sha(abs_path("external/perturbation-drive")),
    }

    logger = RunLogger(run_dir, model_name, str(ckpt) if ckpt else None, "udacity", git_info)

    logger.snapshot_configs(
        sim_app=str(sim_app),
        ckpt=str(ckpt) if ckpt else None,
        cfg_paths=paths_cfg,
        cfg_roads={road_name: spec for road_name, spec in roads.pick()},
        cfg_perturbations=scenarios_by_road,
        cfg_run=run_cfg,
        cfg_host_port={"host": sim_cfg["host"], "port": sim_cfg["port"]},
    )
    logger.snapshot_env(pip_freeze())

    roadgen = CustomRoadGenerator()
    ep_idx = 0

    max_steps = run_cfg.get("max_steps")
    road_cooldown_s = float(run_cfg.get("road_cooldown_s", 3.0))

    show_image = bool(run_cfg.get("show_image", True))

    try:
        for road_name, spec in roads.pick():
            angles = spec["angles"]
            segs = spec["segs"]
            road_scenarios: List[Dict[str, Any]] = scenarios_by_road.get(road_name, [])
            if not road_scenarios:
                print(f"[eval:genroads:generalization][INFO] road '{road_name}' has no scenarios, skipping.")
                continue

            print(f"[eval:genroads:generalization][INFO] road='{road_name}' with {len(angles)} segments ({len(road_scenarios)} scenarios)")

            sim = UdacitySimulator(str(sim_app), sim_cfg["host"], int(sim_cfg["port"]), show_image)

            try:
                sim.connect()
                starting_pos = sim.initial_pos

                track_string = roadgen.generate(starting_pos=starting_pos, angles=angles, seg_lengths=segs)
                road = roadgen.previous_road
                if road is None or not hasattr(road, "road_points"):
                    raise RuntimeError("CustomRoadGenerator did not populate previous_road.")

                waypoints = [(p.x, p.y) for p in road.road_points]

                for scen in (s for s in road_scenarios if s.get("active", False)):
                    ep_idx += 1

                    meta = {
                        "road": road_name,
                        "angles": angles,
                        "segs": segs,
                        "start": {
                            "x": starting_pos[0],
                            "y": starting_pos[1],
                            "yaw_deg": starting_pos[2],
                            "speed": starting_pos[3],
                        },
                        "scenario_name": scen["name"],
                        "scenario": scen,
                        "level": scen["level"],
                        "controller": model_name,
                        "image_size": {
                            "h": run_cfg.get("image_size_hw", [240, 320])[0],
                            "w": run_cfg.get("image_size_hw", [240, 320])[1],
                        },
                        "episodes": 1,
                        "ckpt_name": ckpt_name,
                        "perturbation": scen["name"],
                        "severity": scen["level"],
                    }
                    eid, ep_dir = logger.new_episode(ep_idx, meta)

                    log_file = ep_dir / "pd_log.json"

                    status, wall = run_scenario_episode(
                        sim=sim,
                        controller=adapter,
                        track_string=track_string,
                        waypoints=waypoints,
                        scenario=scen,
                        log_path=log_file,
                        max_steps=max_steps,
                        enable_preview=show_image,
                    )
                    logger.complete_episode(eid, status=status, wall_time_s=wall)

            finally:
                try:
                    sim.tear_down()
                except Exception:
                    pass
                time.sleep(road_cooldown_s)

    except KeyboardInterrupt:
        print("\n[eval:genroads:generalization][WARN] Ctrl-C — stopping.")

    print("[eval:genroads:generalization][INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
