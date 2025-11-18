"""
Perturbs steering/throttle at specified waypoint indices.
"""

from __future__ import annotations

import json
import sys
import time
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import gym

# add project root & perturbation-drive to path
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PD = ROOT / "external" / "perturbation-drive"
if str(PD) not in sys.path:
    sys.path.insert(0, str(PD))

from sims.udacity.maps.configs.run import HOST, PORT
from sims.udacity.maps.genroads.configs import paths, roads, run, scenarios
from sims.udacity.logging.eval_runs import RunLogger, prepare_run_dir, module_public_dict, best_effort_git_sha, pip_freeze
from sims.udacity.adapters.dave2_adapter import Dave2Adapter
from sims.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter

from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from perturbationdrive import ImageCallBack
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
) -> Tuple[Tuple[float, float], Phase, List[str], List[scenarios.Perturbation]]:
    """
    Apply all perturbations that are active at the current waypoint index.

    Returns:
      (steer, throttle), phase, active_controls, active_perturbations
    """
    steer, throttle = base_action
    active_controls: List[str] = []
    active_perturbations: List[scenarios.Perturbation] = []

    for p in scenario.perturbations:
        if not (p.start_wp <= wp_idx <= p.end_wp):
            continue

        active_controls.append(p.control.value)
        active_perturbations.append(p)

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

    steer = max(-1.0, min(1.0, steer))
    throttle = max(-1.0, min(1.0, throttle))

    phase = classify_phase(wp_idx, scenario)
    return (steer, throttle), phase, active_controls, active_perturbations


def _extract_pid_state(info: dict) -> dict:
    """
    Collect everything a PID would want to see, directly from info.
    """
    state = {}
    for key in ("cte", "cte_pid", "angle", "speed"):
        if key in info:
            state[key] = info[key]
    return state


def format_active_perturbations(perts: List[scenarios.Perturbation]) -> str:
    """
    Human-readable summary of what is currently being applied.

    Examples:
      S*0.70 T*1.20
      S+0.20
    """
    if not perts:
        return "none"

    parts: List[str] = []
    for p in perts:
        if p.control == scenarios.Control.STEERING:
            ctrl = "S"
        else:
            ctrl = "T"

        if p.mode == scenarios.Mode.MUL:
            op = f"*{p.factor:.2f}"
        elif p.mode == scenarios.Mode.ADD:
            # include sign
            op = f"{p.factor:+.2f}"
        else:  # SET
            op = f"={p.factor:.2f}"

        parts.append(f"{ctrl}{op}")

    return " ".join(parts)


def run_scenario_episode(
    sim: UdacitySimulator,
    controller,
    track_string: str,
    waypoints: List[Tuple[float, float]],
    scenario: scenarios.Scenario,
    log_path: Path,
    max_steps: int | None = None,
) -> Tuple[str, float]:
    """
    Runs a single scenario episode and writes per-step history to log_path.

    Uses sim.client (UdacityGymEnv_RoadGen) directly and, if enabled,
    shows a live image preview via ImageCallBack.
    """
    if sim.client is None:
        raise RuntimeError("UdacitySimulator.client is None. Did you call sim.connect()?")

    env = sim.client
    history: List[dict] = []

    # Optional preview window
    monitor: ImageCallBack | None = None
    try:
        if getattr(sim, "show_image_cb", False):
            monitor = ImageCallBack()
            monitor.display_waiting_screen()

        # Reset to the generated road; same API as in simulate_scanario
        obs = env.reset(skip_generation=False, track_string=track_string)
        # Initial observe to get info (position etc.)
        obs, done, info = env.observe()

        start_time = time.perf_counter()
        step_idx = 0

        while not done:
            # Model action – adapter handles image preprocessing internally
            model_actions = controller.action(obs)
            model_actions = np.asarray(model_actions, dtype=np.float32)

            # Expect shape (1, 2) or (batch, 2)
            if model_actions.ndim == 1:
                model_actions = model_actions.reshape(1, -1)

            model_steer = float(model_actions[0][0])
            model_throttle = float(model_actions[0][1])

            # Position + waypoint index from info["pos"]
            pos = info.get("pos")

            # Use the first two entries as (x, y) to match road_points (x, y)
            x = float(pos[0])
            y = float(pos[1])

            wp_idx = nearest_wp_index(x, y, waypoints)

            # prints current waypoint index
            print(f"[scenarios:genroads] step={step_idx:04d}  wp_idx={wp_idx:4d}  x={x:7.2f}  y={y:7.2f}", flush=True)

            # Apply scenario
            (act_steer, act_throttle), phase, active_controls, active_perts = apply_scenario(
                scenario, wp_idx, (model_steer, model_throttle)
            )

            actual_actions = np.array([[act_steer, act_throttle]], dtype=np.float32)

            # Clip to action space, same as in simulate_scanario
            if isinstance(env.action_space, gym.spaces.Box):
                actual_actions = np.clip(
                    actual_actions, env.action_space.low, env.action_space.high
                )

            pert_summary = format_active_perturbations(active_perts)

            # Preview image (show what we actually send to the sim)
            if monitor is not None:
                phase_label = phase.value
                scenario_label = f"{scenario.name} (level {scenario.level}):\nRoad: {scenario.road}, Waypoint: {wp_idx}\nPhase: {phase_label} --> Perturbations: {pert_summary}\nComment: {scenario.comment}"
                monitor.display_img(
                    obs,
                    f"{actual_actions[0][0]: .3f}",
                    f"{actual_actions[0][1]: .3f}",
                    scenario_label,
                )

            # Step env; UdacityGymEnv_RoadGen returns (obs, done, info)
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
        with log_path.open("w") as f:
            json.dump({"steps": history}, f, indent=2)
        return "ok", wall

    except Exception as e:
        wall = time.perf_counter() - start_time if "start_time" in locals() else 0.0
        with log_path.open("w") as f:
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

    # store scenario config in cfg_perturbations slot
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
            road_scenarios = scenarios.SCENARIOS.get(road_name, [])
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
                # launches Unity, creates UdacityGymEnv_RoadGen, sets initial_pos
                sim.connect()
                starting_pos = sim.initial_pos

                # Generate road in PerturbationDrive and pull:
                # - track_string for env.reset(...)
                # - road_points for waypoint geometry
                track_string = roadgen.generate(
                    starting_pos=starting_pos,
                    angles=angles,
                    seg_lengths=segs,  # correct keyword
                )
                road = roadgen.previous_road
                if road is None or not hasattr(road, "road_points"):
                    raise RuntimeError(
                        "CustomRoadGenerator did not populate previous_road. "
                        "Check PerturbationDrive version / API."
                    )

                waypoints = [(p.x, p.y) for p in road.road_points]

                for scen in (s for s in road_scenarios if getattr(s, "active", False)):
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
                        "scenario": scen.to_dict(),
                        "level": scen.level,
                        "controller": model_name,
                        "image_size": {"h": run.IMAGE_SIZE[0], "w": run.IMAGE_SIZE[1]},
                        "episodes": 1,
                        "ckpt_name": ckpt_name,
                        "perturbation": scen.name,
                        "severity": scen.level,
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
