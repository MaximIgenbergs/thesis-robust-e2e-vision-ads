"""
Generic Udacity driver using adapters + PerturbationDrive.
Selects model via sims/udacity/configs/run.py (MODEL_NAME)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Union

# add project root & perturbation-drive to path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PD = ROOT / "external" / "perturbation-drive"
if str(PD) not in sys.path:
    sys.path.insert(0, str(PD))

from sims.udacity.configs import paths, roads, perturbations, run
from sims.udacity.utils.logging import RunLogger, make_run_dir, make_run_id, module_public_dict, best_effort_git_sha, pip_freeze
from sims.udacity.adapters.dave2_adapter import Dave2Adapter
from perturbationdrive import Scenario, PerturbationDrive
try:
    from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
except ImportError:
    from perturbationdrive import CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator


# ---- helpers ----

def _abs(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _make_scenarios(waypoints, pert_names: List[str], severities: List[int], episodes: int) -> List[Scenario]:
    scens: List[Scenario] = []
    for _ in range(episodes):
        for p in pert_names:
            for s in severities:
                scens.append(Scenario(waypoints=waypoints, perturbation_function=p, perturbation_scale=int(s)))
    return scens


def _build_adapter(model_name: str, image_size_hw, ckpt_path: Path | None):
    if model_name == "dave2":
        return Dave2Adapter(weights=ckpt_path, image_size_hw=image_size_hw, device=None, normalize="imagenet")
    # elif model_name == "dave2_gru":
    #     from sims.udacity.adapters.dave2_gru_adapter import Dave2GRUAdapter
    #     return Dave2GRUAdapter(weights=ckpt_path, image_size_hw=image_size_hw, device=None, normalize="imagenet")
    raise ValueError(f"Unknown MODEL_NAME '{model_name}' in sims.udacity.configs.run")


def main() -> int:
    sim_app = _abs(paths.PD_SIM)
    ckpt = _abs(paths.DAVE2_CKPT) if getattr(paths, "DAVE2_CKPT", None) else None

    if not sim_app.exists():
        raise FileNotFoundError(f"PD_SIM not found: {sim_app}\nEdit sims/udacity/configs/paths.py")
    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"DAVE2_CKPT not found: {ckpt}\nEdit sims/udacity/configs/paths.py or download the file.")

    model_name = getattr(run, "MODEL_NAME", "dave2")
    runs_root = _abs(paths.RUNS_DIR) / "robustness" / model_name
    adapter = _build_adapter(model_name, image_size_hw=run.IMAGE_SIZE, ckpt_path=ckpt)

    # ---- logging ----
    ckpt_name = ckpt.stem if ckpt is not None else model_name
    run_id = make_run_id(ckpt_name)
    run_dir = make_run_dir(runs_root, run_id)
    print(f"[drive] model={model_name} logs -> {run_dir}")

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
    logger.snapshot_configs(
        sim_app=sim_app,
        ckpt=ckpt,
        cfg_paths=module_public_dict(paths),
        cfg_roads=module_public_dict(roads),
        cfg_perturbations=module_public_dict(perturbations),
        cfg_run=module_public_dict(run),
    )
    logger.snapshot_env(pip_freeze())

    # ---- run ----
    chunks = perturbations.CHUNKS if len(perturbations.CHUNKS) > 0 else [perturbations.LIST]

    roadgen = CustomRoadGenerator()
    ep_idx = 0

    try:
        for road_name, spec in roads.pick():
            angles = spec["angles"]
            segs = spec["segs"]
            print(f"[drive] road '{road_name}' ({len(angles)} segments)")

            sim = UdacitySimulator(
                simulator_exe_path=str(sim_app),
                host=run.HOST,
                port=run.PORT,
                show_image_cb=run.SHOW_IMAGE,
            )
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

                if run.RECONNECT:
                    # Reconnect for each severity (isolated sessions)
                    for chunk in chunks:
                        for pert in chunk:
                            for sev in run.SEVERITIES:
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
                                    "image_size": {"h": run.IMAGE_SIZE[0], "w": run.IMAGE_SIZE[1]},
                                    "episodes": int(run.EPISODES),
                                    "ckpt_name": ckpt_name,
                                }
                                eid, ep_dir = logger.new_episode(ep_idx, meta)
                                log_file = ep_dir / "pd_log.json"

                                scens = _make_scenarios(waypoints, [pert], [sev], run.EPISODES)
                                t0 = time.perf_counter()
                                try:
                                    bench.simulate_scenarios(
                                        scenarios=scens,
                                        log_dir=str(log_file),
                                        image_size=run.IMAGE_SIZE,
                                    )
                                    logger.complete_episode(eid, status="ok", wall_time_s=(time.perf_counter() - t0))
                                except Exception as e:
                                    logger.complete_episode(eid, status=f"error:{type(e).__name__}", wall_time_s=(time.perf_counter() - t0))
                                    raise
                                # rotate session
                                try:
                                    bench.simulator.tear_down()
                                except Exception:
                                    pass
                                time.sleep(8)
                                bench.simulator.connect()
                                adapter.reset()
                else:
                    # Single session for all severities
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
                                "severity": "all",
                                "image_size": {"h": run.IMAGE_SIZE[0], "w": run.IMAGE_SIZE[1]},
                                "episodes": int(run.EPISODES),
                                "ckpt_name": ckpt_name,
                            }
                            eid, ep_dir = logger.new_episode(ep_idx, meta)
                            log_file = ep_dir / "pd_log.json"

                            scens = _make_scenarios(waypoints, [pert], run.SEVERITIES, run.EPISODES)
                            t0 = time.perf_counter()
                            try:
                                bench.simulate_scenarios(
                                    scenarios=scens,
                                    log_dir=str(log_file),
                                    image_size=run.IMAGE_SIZE,
                                )
                                logger.complete_episode(eid, status="ok", wall_time_s=(time.perf_counter() - t0))
                            except Exception as e:
                                logger.complete_episode(eid, status=f"error:{type(e).__name__}", wall_time_s=(time.perf_counter() - t0))
                                raise

            finally:
                try:
                    bench.simulator.tear_down()
                except Exception:
                    pass
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n[drive] Ctrl-C — stopping.")
    print("[drive] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
