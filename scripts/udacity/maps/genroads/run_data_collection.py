"""
Collect training data on genroads using PerturbationDrive.

Creates a pid_YYYYMMDD-HHMMSS/ run directory under DATA_DIR and fills it with image_*.jpg / record_*.json pairs converted from PerturbationDrive logs.
"""

from __future__ import annotations
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.udacity.maps.genroads.utils.load_roads import load_roads
from scripts.udacity.logging.data_collection_runs import make_run_dir, convert_outputs

from perturbationdrive import PerturbationDrive
from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from scripts import abs_path

SIM_PATH = "/home/maxim/thesis-robust-e2e-vision-ads/binaries/genroads/udacity_linux/udacity_binary.x86_64"
HOST = "127.0.0.1"
PORT = 9091

DATA_DIR = "/home/maxim/thesis-robust-e2e-vision-ads/data/genroads"
ROADS_PATH = abs_path("scripts/udacity/maps/genroads/roads/roads.yaml")
ROADS_SET = "data_collection"


def main() -> None:
    data_root = Path(DATA_DIR).expanduser().resolve()
    run_dir = make_run_dir(data_root, prefix="pid")
    raw_pd_dir = run_dir / "raw_pd_logs"
    raw_pd_dir.mkdir(parents=True, exist_ok=True)

    sim_app = abs_path(SIM_PATH)
    if not sim_app.exists():
        raise FileNotFoundError(f"SIM not found: {sim_app}")

    print(f"[scripts:genroads:collection] run_dir: {run_dir}")

    # Load roads + sets from YAML
    roads_def, road_sets = load_roads(ROADS_PATH)

    print(f"[scripts:genroads:collection] roads_set: {ROADS_SET}")

    if ROADS_SET not in road_sets:
        raise KeyError(f"Unknown road set '{ROADS_SET}' in {ROADS_PATH}. Known sets: {list(road_sets.keys())}")

    selected_roads = list(road_sets[ROADS_SET])

    for road_name in selected_roads:
        spec = roads_def[road_name]
        angles = spec["angles"]
        segs = spec["segs"]

        print(f"[scripts:genroads:collection] road: {road_name} segments={len(angles)}")

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

    convert_outputs(pd_logs_dir=raw_pd_dir, out_pairs_dir=run_dir)
    print(f"[scripts:genroads:collection] done. pairs in: {run_dir}")


if __name__ == "__main__":
    main()
