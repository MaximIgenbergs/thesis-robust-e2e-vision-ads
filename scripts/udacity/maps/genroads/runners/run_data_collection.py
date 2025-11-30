"""
Collect nominal training data on genroads using PerturbationDrive.

Creates a pid_YYYYMMDD-HHMMSS/ run directory under DATA_DIR and fills it with image_*.jpg / record_*.json pairs converted from PerturbationDrive logs.
"""

from __future__ import annotations
import time
from pathlib import Path

from scripts.udacity.maps.genroads.configs import paths, roads
from scripts.udacity.logging.data_collection import make_run_dir, convert_outputs
from scripts import abs_path

from perturbationdrive import PerturbationDrive
from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator

HOST = "127.0.0.1"
PORT = 9091


def main() -> None:
    data_root = Path(paths.DATA_DIR).expanduser().resolve()
    run_dir = make_run_dir(data_root, prefix="pid")
    raw_pd_dir = run_dir / "raw_pd_logs"
    raw_pd_dir.mkdir(parents=True, exist_ok=True)

    sim_app = abs_path(getattr(paths, "SIM", getattr(paths, "SIM", "")))
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}\n Edit scripts/udacity/maps/genroads/configs/paths.py")

    print(f"[scripts:genroads:collection] run_dir: {run_dir}")

    roads_set = "data_collection"
    print(f"[scripts:genroads:collection] roads_set: {roads_set}")

    try:
        for road_name, spec in roads.pick(roads_set):
            angles = spec["angles"]
            segs = spec["segs"]

            print(
                f"[scripts:genroads:collection] road: {road_name} "
                f"segments={len(angles)}"
            )

            sim = UdacitySimulator(simulator_exe_path=str(sim_app), host=HOST, port=PORT)
            bench = PerturbationDrive(simulator=sim, ads=None)

            try:
                roadgen = CustomRoadGenerator(num_control_nodes=len(angles))

                ts = time.strftime("%Y_%m_%d_%H_%M_%S")
                log_file = raw_pd_dir / f"udacity_road_{road_name}_logs_{ts}.json"

                bench.grid_seach(
                    perturbation_functions=[],
                    attention_map={},
                    road_generator=roadgen,
                    road_angles=angles,
                    road_segments=segs,
                    log_dir=str(log_file),
                    overwrite_logs=True,
                    image_size=(240, 320),
                    test_model=False,
                    collect_train_data=True,
                    perturb=False,
                )

            finally:
                try:
                    bench.simulator.tear_down()
                except Exception:
                    pass
                time.sleep(3.0)

    except KeyboardInterrupt:
        print("\n[scripts:genroads:collection] Ctrl-C — interrupted. Converting collected logs.")
    finally:
        convert_outputs(pd_logs_dir=raw_pd_dir, out_pairs_dir=run_dir)
        print(f"[scripts:genroads:collection] done. pairs in: {run_dir}")


if __name__ == "__main__":
    main()
