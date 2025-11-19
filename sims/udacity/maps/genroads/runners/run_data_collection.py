# sims/udacity/maps/genroads/runners/run_data_collection.py
"""
Collect nominal training data on 'genroads' using PerturbationDrive over a curated
set of roads (roads.SETS['data_collection']). For each road:
  - Launch Unity once (visible UI)
  - Let PD grid_seach() connect, build waypoints, and drive (no perturbations)
  - Tear down Unity before moving to the next road

Outputs:
  <DATA_DIR>/pid_YYYYMMDD-HHMMSS/
    raw_pd_logs/
      udacity_road_<roadname>_logs_<timestamp>.json
      udacity_road_<roadname>_logs_<timestamp>___0_original/*.jpg
    image_000001.jpg
    record_000001.json
    ...

SIM and DATA_DIR are taken from sims/udacity/maps/genroads/configs/paths.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Union

# add project root & perturbation-drive to path
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PD_DIR = ROOT / "external" / "perturbation-drive"
if str(PD_DIR) not in sys.path:
    sys.path.insert(0, str(PD_DIR))

from sims.udacity.maps.configs.run import HOST, PORT
from sims.udacity.maps.genroads.configs import paths as gen_paths, roads
from sims.udacity.logging.data_collection import make_run_dir, convert_outputs

from perturbationdrive import PerturbationDrive
from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator

# ---- helpers ----

def _abs(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()

def _udacity_sim_path() -> Path:
    sim_path = _abs(getattr(gen_paths, "SIM", getattr(gen_paths, "SIM", "")))
    if not sim_path.exists():
        raise FileNotFoundError(f"SIM not found: {sim_path}\nEdit sims/udacity/maps/genroads/configs/paths.py")
    return sim_path

# ---- main ----

def collect_genroads() -> None:
    # Timestamped run dir under DATA_DIR; keep raw PD logs for traceability.
    data_root = Path(gen_paths.DATA_DIR).expanduser().resolve()
    run_dir   = make_run_dir(data_root, prefix="pid")
    raw_pd_dir = run_dir / "raw_pd_logs"
    raw_pd_dir.mkdir(parents=True, exist_ok=True)

    sim_exe = str(_udacity_sim_path())

    print(f"[collect:genroads] run_dir={run_dir}")
    print("[collect:genroads] roads set='data_collection'") # TODO: what is that. is there a config missing? 

    # Iterate deterministic data-collection roads
    for road_name, spec in roads.pick("data_collection"):
        angles = spec["angles"]
        segs   = spec["segs"]
        print(f"{'#'*5} Road {road_name} ({len(angles)} segments) {'#'*5}")

        # Create a fresh simulator per road (visible window), but DO NOT connect here.
        # grid_seach() will connect and also tear_down() at the end.
        sim = UdacitySimulator(
            simulator_exe_path=sim_exe,
            host=HOST,
            port=PORT,
        )
        bench = PerturbationDrive(simulator=sim, ads=None)  # ADS=None → PID controller inside PD

        try:
            # Let grid_seach() handle connect() and waypoint generation using initial_pos.
            roadgen = CustomRoadGenerator(num_control_nodes=len(angles))

            ts = time.strftime("%Y_%m_%d_%H_%M_%S")
            log_file = raw_pd_dir / f"udacity_road_{road_name}_logs_{ts}.json"

            bench.grid_seach(
                perturbation_functions=[],   # nominal
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
            # grid_seach() already calls self.simulator.tear_down() internally,
            # but we'll be defensive in case of exceptions mid-run.
            try:
                bench.simulator.tear_down()
            except Exception:
                pass
            # Give the process a moment to exit cleanly and release the port.
            time.sleep(3.0)

    # Convert all PD logs → flat pairs in run_dir
    convert_outputs(pd_logs_dir=raw_pd_dir, out_pairs_dir=run_dir)

    print(f"[collect:genroads] done. pairs in: {run_dir}")

if __name__ == "__main__":
    collect_genroads()
