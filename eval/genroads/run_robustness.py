"""
Robustness evaluation of a model on the Udacity genroads map using image perturbations from PerturbationDrive over a set of generated roads.

Arguments:
    --model MODEL_NAME
        Override experiment.default_model from cfg_robustness.yaml.
        Examples: --model dave2, --model dave2_gru, --model vit
"""

from __future__ import annotations
import argparse
import time
from typing import List, Dict, Any

from scripts.udacity.logging.eval_runs import RunLogger, prepare_run_dir, best_effort_git_sha, pip_freeze
from scripts.udacity.adapters.utils.build_adapter import build_adapter
from scripts.udacity.maps.genroads.roads.load_roads import load_roads
from scripts import abs_path, load_cfg

from perturbationdrive import Scenario, PerturbationDrive
from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator


def make_scenarios(waypoints, pert_names: List[str], severities: List[int], episodes: int) -> List[Scenario]:
    """
    Build a flat list of PerturbationDrive scenarios for (perturbation, severity) pairs.
    """
    scenarios: List[Scenario] = []
    for _ in range(episodes):
        for p in pert_names:
            for s in severities:
                scenarios.append(Scenario(waypoints=waypoints, perturbation_function=p, perturbation_scale=int(s)))
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=("Override experiment.default_model from eval/genroads/cfg_robustness.yaml.\nExamples: --model dave2, --model dave2_gru, --model vit"),
    )
    args = parser.parse_args()

    cfg = load_cfg("eval/genroads/cfg_robustness.yaml")

    udacity_cfg = cfg["udacity"]
    models_cfg = cfg["models"]
    run_cfg = cfg["run"]
    logging_cfg = cfg["logging"]
    pert_cfg = cfg["perturbations"]
    baseline = bool(pert_cfg.get("baseline", False))

    roads_yaml = abs_path("scripts/udacity/maps/genroads/roads/roads.yaml") # TODO: maybe not hardcode this path?
    roads_def, road_sets = load_roads(roads_yaml)

    road_set_id = run_cfg.get("road_set")
    if not road_set_id or road_set_id not in road_sets:
        raise KeyError(f"run.road_set='{road_set_id}' is invalid in eval/genroads/cfg_robustness.yaml.\nKnown sets: {list(road_sets.keys())}")
    selected_roads: List[str] = list(road_sets[road_set_id])

    model_defs = {k: v for k, v in models_cfg.items() if k != "default_model"}

    model_name = args.model or models_cfg.get("default_model", "dave2")
    if model_name not in model_defs:
        raise ValueError(f"Model '{model_name}' not defined under models in eval/genroads/cfg_robustness.yaml")

    adapter, ckpt = build_adapter(model_name, model_defs[model_name])

    sim_app = abs_path(udacity_cfg["binary"])
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit udacity.binary in eval/genroads/cfg_robustness.yaml")

    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"{model_name.upper()} checkpoint not found: {ckpt}\nEdit models.{model_name}.checkpoint in eval/genroads/cfg_robustness.yaml")

    runs_root = abs_path(logging_cfg["runs_dir"])

    ckpt_name = ckpt.stem if ckpt is not None else model_name
    map_name = udacity_cfg.get("map", "genroads")

    _, run_dir = prepare_run_dir(model_name=model_name, runs_root=runs_root)
    print(f"[eval:genroads:robustness][INFO] model={model_name} logs -> {run_dir}")

    include_git = logging_cfg.get("include_git_sha", {})
    git_info: Dict[str, Any] = {}

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
            cfg_roads={"map": map_name, "roads_yaml": str(roads_yaml), "road_set": road_set_id, "roads": selected_roads},
            cfg_perturbations=pert_cfg,
            cfg_run=run_cfg,
            cfg_host_port={"host": udacity_cfg["host"], "port": udacity_cfg["port"]},
        )

    if logging_cfg.get("snapshot_env", True):
        logger.snapshot_env(pip_freeze())

    chunks: list[list[str]] = pert_cfg.get("chunks")
    severities = list(pert_cfg.get("severities", [1, 2, 3, 4]))
    episodes = int(pert_cfg.get("episodes", 1))

    image_size_hw = tuple(run_cfg.get("image_size_hw", [240, 320]))
    show_image = bool(run_cfg.get("show_image", True))
    reconnect = bool(run_cfg.get("reconnect", True))
    reconnect_cooldown_s = float(run_cfg.get("reconnect_cooldown_s", 8.0))
    road_cooldown_s = float(run_cfg.get("road_cooldown_s", 3.0))

    roadgen = CustomRoadGenerator(num_control_nodes=10)
    ep_idx = 0

    try:
        for road_name in selected_roads:
            spec = roads_def[road_name]
            angles = spec["angles"]
            segs = spec["segs"]
            print(f"[eval:genroads:robustness][INFO] road='{road_name}' ({len(angles)} segments)")

            sim = UdacitySimulator(str(sim_app), udacity_cfg["host"], int(udacity_cfg["port"]), show_image)
            bench = PerturbationDrive(simulator=sim, ads=adapter)

            try:
                bench.simulator.connect()
                starting_pos = bench.simulator.initial_pos
                waypoints = roadgen.generate(
                    starting_pos=starting_pos,
                    angles=angles,
                    seg_length=segs,
                )

                adapter.reset()

                if baseline:
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
                        "perturbation": None,
                        "severity": 0,
                        "image_size": {"h": image_size_hw[0], "w": image_size_hw[1]},
                        "episodes": 1,
                        "ckpt_name": ckpt_name,
                    }
                    eid, ep_dir = logger.new_episode(ep_idx, meta)
                    log_file = ep_dir / "log.json"

                    scens = [Scenario(waypoints=waypoints, perturbation_function=None, perturbation_scale=0)]
                    t0 = time.perf_counter()
                    try:
                        bench.simulate_scenarios(scenarios=scens, log_dir=str(log_file), image_size=image_size_hw)
                        logger.complete_episode(eid, status="ok", wall_time_s=time.perf_counter() - t0)
                    except Exception as e:
                        logger.complete_episode(eid, status=f"error:{type(e).__name__}", wall_time_s=time.perf_counter() - t0)
                        raise

                if reconnect: # Reconnect for each run
                    for chunk in chunks:
                        for pert in chunk:
                            for sev in severities:
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
                                    "perturbation": pert,
                                    "severity": int(sev),
                                    "image_size": {"h": image_size_hw[0], "w": image_size_hw[1]},
                                    "episodes": int(episodes),
                                    "ckpt_name": ckpt_name,
                                }
                                eid, ep_dir = logger.new_episode(ep_idx, meta)
                                log_file = ep_dir / "log.json"

                                scens = make_scenarios(waypoints, [pert], [sev], episodes)
                                t0 = time.perf_counter()
                                try:
                                    bench.simulate_scenarios(scenarios=scens, log_dir=str(log_file), image_size=image_size_hw)
                                    logger.complete_episode(eid, status="ok", wall_time_s=time.perf_counter() - t0)
                                except Exception as e:
                                    logger.complete_episode(eid, status=f"error:{type(e).__name__}", wall_time_s=time.perf_counter() - t0)
                                    raise

                                try:
                                    bench.simulator.tear_down()
                                except Exception:
                                    pass
                                time.sleep(reconnect_cooldown_s)
                                bench.simulator.connect()
                                adapter.reset()
                else: # Single session for all severities
                    for chunk in chunks:
                        for pert in chunk:
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
                                "perturbation": pert,
                                "severity": severities,
                                "image_size": {"h": image_size_hw[0], "w": image_size_hw[1]},
                                "episodes": int(episodes),
                                "ckpt_name": ckpt_name,
                            }
                            eid, ep_dir = logger.new_episode(ep_idx, meta)
                            log_file = ep_dir / "log.json"

                            scens = make_scenarios(waypoints, [pert], severities, episodes)
                            t0 = time.perf_counter()
                            try:
                                bench.simulate_scenarios(scenarios=scens, log_dir=str(log_file), image_size=image_size_hw)
                                logger.complete_episode(eid, status="ok", wall_time_s=(time.perf_counter() - t0))
                            except Exception as e:
                                logger.complete_episode(eid, status=f"error:{type(e).__name__}", wall_time_s=(time.perf_counter() - t0))
                                raise
            finally:
                try:
                    bench.simulator.tear_down()
                except Exception:
                    pass
                time.sleep(road_cooldown_s)

    except KeyboardInterrupt:
        print("\n[eval:genroads:robustness][WARN] Ctrl-C — stopping.")

    print("[eval:genroads:robustness][INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
