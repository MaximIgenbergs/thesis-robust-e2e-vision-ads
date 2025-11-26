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
from pathlib import Path
from typing import List, Dict, Any
import yaml

from scripts.udacity.logging.eval_runs import RunLogger, prepare_run_dir, best_effort_git_sha, pip_freeze
from scripts.udacity.adapters.dave2_adapter import Dave2Adapter
from scripts.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter
from scripts.udacity.maps.genroads.configs import roads
from scripts import abs_path

from perturbationdrive import Scenario, PerturbationDrive
from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator


def load_cfg() -> Dict[str, Any]:
    cfg_path = Path(__file__).with_name("cfg_robustness.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_adapter(model_name: str, model_cfg: dict, ckpts_dir: Path):
    """
    Instantiate the correct Udacity adapter and return (adapter, ckpt_path).
    """
    ckpt_rel = model_cfg.get("checkpoint")
    ckpt = abs_path(ckpts_dir / ckpt_rel) if ckpt_rel else None

    image_size_hw = tuple(model_cfg.get("image_size_hw", [240, 320]))
    normalize = model_cfg.get("normalize", "imagenet")

    if model_name == "dave2":
        return (Dave2Adapter(weights=ckpt, image_size_hw=image_size_hw, device=None, normalize=normalize), ckpt)

    if model_name == "dave2_gru":
        seq_len = int(model_cfg.get("sequence_length", 3))
        return (Dave2GRUAdapter(weights=ckpt, image_size_hw=image_size_hw, seq_len=seq_len, device=None, normalize=normalize), ckpt)

    raise ValueError(f"Unknown model '{model_name}' in cfg_robustness.yaml")


def make_scenarios(waypoints, pert_names: List[str], severities: List[int], episodes: int) -> List[Scenario]:
    """
    Build a flat list of PerturbationDrive scenarios for (perturbation, severity) pairs.
    """
    scenarios: List[Scenario] = []
    for _ in range(episodes):
        for p in pert_names:
            for s in severities:
                scenarios.append(
                    Scenario(
                        waypoints=waypoints,
                        perturbation_function=p,
                        perturbation_scale=int(s),
                    )
                )
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override experiment.default_model from cfg_robustness.yaml. "
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
        raise ValueError(f"Model '{model_name}' not defined under models in cfg_robustness.yaml")

    ckpts_dir = abs_path(paths_cfg["ckpts_dir"])
    runs_root = abs_path(paths_cfg["runs_dir"])
    data_dir = abs_path(paths_cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    adapter, ckpt = build_adapter(model_name, models_cfg[model_name], ckpts_dir)

    sim_app = abs_path(sim_cfg["binary"])
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\nEdit sim.binary in cfg_robustness.yaml")

    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(
            f"{model_name.upper()} checkpoint not found: {ckpt}\n"
            "Edit models.*.checkpoint in cfg_robustness.yaml"
        )

    ckpt_name = ckpt.stem if ckpt is not None else model_name
    _, run_dir = prepare_run_dir(
        map_name=exp_cfg.get("map", "genroads"),
        test_type=exp_cfg.get("test_type", "robustness"),
        model_name=model_name,
        tag=ckpt_name,
    )
    print(f"[eval:genroads:robustness][INFO] model={model_name} logs -> {run_dir}")

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
        cfg_roads={"map": exp_cfg.get("map", "genroads"), "selector": "roads.pick()"},
        cfg_perturbations=pert_cfg,
        cfg_run=run_cfg,
        cfg_host_port={"host": sim_cfg["host"], "port": sim_cfg["port"]},
    )
    logger.snapshot_env(pip_freeze())

    chunks: list[list[str]] = pert_cfg.get("chunks") or [pert_cfg["list"]]
    severities = list(pert_cfg.get("severities", [1, 2, 3, 4]))
    episodes = int(pert_cfg.get("episodes", 1))

    image_size_hw = tuple(run_cfg.get("image_size_hw", [240, 320]))
    show_image = bool(run_cfg.get("show_image", True))
    reconnect = bool(run_cfg.get("reconnect", True))
    reconnect_cooldown_s = float(run_cfg.get("reconnect_cooldown_s", 8.0))
    road_cooldown_s = float(run_cfg.get("road_cooldown_s", 3.0))

    roadgen = CustomRoadGenerator()
    ep_idx = 0

    try:
        for road_name, spec in roads.pick():
            angles = spec["angles"]
            segs = spec["segs"]
            print(f"[eval:genroads:robustness][INFO] road='{road_name}' ({len(angles)} segments)")

            sim = UdacitySimulator(str(sim_app), sim_cfg["host"], int(sim_cfg["port"]), show_image)
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

                                # rotate session
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
