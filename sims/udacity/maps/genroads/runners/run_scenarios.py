# sims/udacity/maps/genroads/runners/run_scenarios.py

"""
Scenario-based experiments on genroads.

- Builds roads from angles + segs (CustomRoadGenerator).
- Runs one controller (model or PID) per scenario.
- Perturbs steering/throttle at specified waypoint indices.
- Logs model vs actual actions and a phase tag:
  "before_setup", "setup", "reaction".
"""

from __future__ import annotations

import json
import sys
import time
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

# project root on path
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PD = ROOT / "external" / "perturbation-drive"
if str(PD) not in sys.path:
    sys.path.insert(0, str(PD))

from sims.udacity.maps.configs.run import HOST, PORT
from sims.udacity.maps.configs import perturbations  # for EPISODES if you want to reuse that
from sims.udacity.maps.genroads.configs import paths, roads, run, scenarios
from sims.udacity.logging.eval_runs import (
    RunLogger,
    prepare_run_dir,
    module_public_dict,
    best_effort_git_sha,
    pip_freeze,
)
from sims.udacity.adapters.dave2_adapter import Dave2Adapter
from sims.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter

try:
    from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
except ImportError:
    from perturbationdrive import CustomRoadGenerator

from examples.udacity.udacity_simulator import UdacitySimulator


def _abs(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _build_adapter(model_name: str, image_size_hw, ckpt_path: Path | None):
    if model_name == "dave2":
        return Dave2Adapter(weights=ckpt_path, image_size_hw=image_size_hw, device=None, normalize="imagenet")
    if model_name == "dave2_gru":
        return Dave2GRUAdapter(weights=ckpt_path, image_size_hw=image_size_hw, device=None, normalize="imagenet")
    raise ValueError(f"Unknown MODEL_NAME '{model_name}' in sims/udacity/configs/genroads/run.py")


def nearest_wp_index(x: float, y: float, waypoints: List[Tuple[float, float]]) -> int:
    """Simple nearest-neighbour lookup in waypoint space."""
    best_i = 0
    best_d2 = float("inf")
    for i, w in enumerate(waypoints):
        dx = w[0] - x
        dy = w[1] - y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_i = i
            best_d2 = d2
    return best_i


class Phase(str, Enum):
    BEFORE = "before_setup"
    SETUP = "setup"
    REACTION = "reaction"


def classify_phase(wp_idx: int, scenario: scenarios.Scenario) -> Phase:
    start_wp, end_wp = scenarios.scenario_bounds(scenario)
    if wp_idx < start_wp:
        return Phase.BEFORE
    if wp_idx <= end_wp:
        return Phase.SETUP
    return Phase.REACTION


def apply_scenario(
    scenario: scenarios.Scenario,
    wp_idx: int,
    base_action: Tuple[float, float],
) -> Tuple[Tuple[float, float], Phase, List[str]]:
    """
    Apply all perturbations that are active at the current waypoint index.

    Returns:
      (steer, throttle), phase, active_controls
    """
    steer, throttle = base_action
    active_controls: List[str] = []

    for p in scenario.perturbations:
        if not (p.start_wp <= wp_idx <= p.end_wp):
            continue

        active_controls.append(p.control.value)

        if p.control == scenarios.Control.STEERING:
            if p.mode == scenarios.Mode.MUL:
                steer *= p.factor
            elif p.mode == scenarios.Mode.ADD:
                steer += p.factor
            elif p.mode == scenarios.Mode.SET:
                steer = p.factor

        elif p.control == scenarios.Control.THROTTLE:
            if p.mode == scenarios.Mode.MUL:
                throttle *= p.factor
            elif p.mode == scenarios.Mode.ADD:
                throttle += p.factor
            elif p.mode == scenarios.Mode.SET:
                throttle = p.factor

    # clamp to [-1, 1] for both (throttle < 0 == braking)
    steer = max(-1.0, min(1.0, steer))
    throttle = max(-1.0, min(1.0, throttle))

    phase = classify_phase(wp_idx, scenario)
    return (steer, throttle), phase, active_controls


def _extract_position(obs, info) -> Tuple[float, float]:
    """
    Helper to get car x,y from obs/info.

    Adjust this to whatever your UdacitySimulator returns.
    """
    if hasattr(obs, "x") and hasattr(obs, "y"):
        return float(obs.x), float(obs.y)
    if isinstance(obs, dict) and "x" in obs and "y" in obs:
        return float(obs["x"]), float(obs["y"])
    if "x" in info and "y" in info:
        return float(info["x"]), float(info["y"])
    raise RuntimeError("Could not extract (x, y) from observation/info.")


def _extract_pid_state(obs, info) -> dict:
    """
    Collect everything a PID might care about (cte, yaw_error, speed, ...).
    Adapt field names to your env.
    """
    state = {}
    for key in ("cte", "yaw_error", "heading_error", "speed"):
        if key in info:
            state[key] = info[key]
    return state


def run_scenario_episode(
    sim: UdacitySimulator,
    controller,
    waypoints,
    scenario: scenarios.Scenario,
    log_path: Path,
    max_steps: int | None = None,
) -> Tuple[str, float]:
    """
    Runs a single scenario episode and writes per-step history to log_path.
    """
    history: List[dict] = []

    # Reset the sim. Adjust this to your actual API.
    obs = sim.reset()
    done = False
    step_idx = 0

    start_time = time.perf_counter()

    try:
        while not done:
            # controller output (model or PID)
            model_steer, model_throttle = controller.action(obs)

            # position and waypoint index
            info_before = {}
            x, y = _extract_position(obs, info_before)
            wp_idx = nearest_wp_index(x, y, waypoints)

            # apply scenario
            (act_steer, act_throttle), phase, active_controls = apply_scenario(
                scenario, wp_idx, (model_steer, model_throttle)
            )

            # step sim (Gym-like assumed; adapt if needed)
            obs_next, reward, done, info = sim.step((act_steer, act_throttle))

            pid_state = _extract_pid_state(obs_next, info)

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
                    "reward": float(reward) if reward is not None else None,
                    "info": info,
                }
            )

            obs = obs_next
            step_idx += 1

            if max_steps is not None and step_idx >= max_steps:
                break

    except Exception as e:
        wall = time.perf_counter() - start_time
        with log_path.open("w") as f:
            json.dump({"steps": history, "error": str(e)}, f, indent=2)
        return f"error:{type(e).__name__}", wall

    wall = time.perf_counter() - start_time
    with log_path.open("w") as f:
        json.dump({"steps": history}, f, indent=2)

    return "ok", wall


def main() -> int:
    sim_app = _abs(paths.SIM)
    model_name = getattr(run, "MODEL_NAME", "dave2")

    if model_name == "dave2":
        ckpt = _abs(paths.DAVE2_CKPT) if getattr(paths, "DAVE2_CKPT", None) else None
    elif model_name == "dave2_gru":
        ckpt = _abs(paths.DAVE2_GRU_CKPT) if getattr(paths, "DAVE2_GRU_CKPT", None) else None
    else:
        raise ValueError(f"Unknown MODEL_NAME '{model_name}' in sims/udacity/configs/genroads/run.py")

    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit sims/udacity/configs/genroads/paths.py")
    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"{model_name.upper()}_CKPT not found: {ckpt}\nEdit sims/udacity/configs/genroads/paths.py")

    adapter = _build_adapter(model_name, image_size_hw=run.IMAGE_SIZE, ckpt_path=ckpt)

    ckpt_name = ckpt.stem if ckpt is not None else model_name
    _, run_dir = prepare_run_dir(
        map_name="genroads",
        test_type="scenarios",
        model_name=model_name,
        tag=ckpt_name,
    )
    print(f"[scenarios:genroads] model={model_name} logs -> {run_dir}")

    git_info = {
        "thesis_sha": best_effort_git_sha(ROOT),
        "perturbation_drive_sha": best_effort_git_sha(PD),
    }
    logger = RunLogger(
        run_dir=run_dir,
        model=model_name,
        checkpoint=(str(ckpt) if ckpt else None),
        sim_name="udacity",
        git_info=git_info,
    )

    # NOTE: eval_runs.RunLogger expects cfg_perturbations here.
    # We just store the scenario config in that slot.
    logger.snapshot_configs(
        sim_app=sim_app,
        ckpt=ckpt,
        cfg_paths=module_public_dict(paths),
        cfg_roads=module_public_dict(roads),
        cfg_perturbations=module_public_dict(scenarios),
        cfg_run=module_public_dict(run),
        cfg_host_port={"host": HOST, "port": PORT},
    )
    logger.snapshot_env(pip_freeze())

    roadgen = CustomRoadGenerator()
    ep_idx = 0

    try:
        for road_name, spec in roads.pick():
            angles = spec["angles"]
            segs = spec["segs"]
            road_scenarios = scenarios.SCENARIOS_BY_ROAD.get(road_name, [])
            if not road_scenarios:
                print(f"[scenarios:genroads] road '{road_name}' has no scenarios, skipping.")
                continue

            print(f"[scenarios:genroads] road '{road_name}' ({len(angles)} segments), {len(road_scenarios)} scenarios")

            sim = UdacitySimulator(
                simulator_exe_path=str(sim_app),
                host=HOST,
                port=PORT,
                show_image_cb=run.SHOW_IMAGE,
            )

            try:
                sim.connect()
                starting_pos = sim.initial_pos

                # waypoints; adjust indexing if your waypoints carry more fields
                wps_raw = roadgen.generate(
                    starting_pos=starting_pos,
                    angles=angles,
                    seg_length=segs,
                )
                waypoints = [(w[0], w[1]) for w in wps_raw]  # just x,y for nearest_wp_index

                for scen in road_scenarios:
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
                        "scenario_name": scen.name,
                        "scenario": scen.to_dict(),        # full scenario content
                        "level": scen.level,
                        "controller": model_name,           # or "pid" if you swap controller
                        "image_size": {"h": run.IMAGE_SIZE[0], "w": run.IMAGE_SIZE[1]},
                        "episodes": 1,
                        "ckpt_name": ckpt_name,
                        # these two are for RunLogger.new_episode manifest entries
                        "perturbation": scen.name,
                        "severity": scen.level,
                    }
                    eid, ep_dir = logger.new_episode(ep_idx, meta)

                    # eval_runs.RunLogger assumes the log file is named 'pd_log.json'
                    log_file = ep_dir / "pd_log.json"

                    status, wall = run_scenario_episode(
                        sim=sim,
                        controller=adapter,
                        waypoints=waypoints,
                        scenario=scen,
                        log_path=log_file,
                        max_steps=None,
                    )
                    logger.complete_episode(eid, status=status, wall_time_s=wall)

            finally:
                try:
                    sim.tear_down()
                except Exception:
                    pass
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n[scenarios:genroads] Ctrl-C — stopping.")

    print("[scenarios:genroads] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
