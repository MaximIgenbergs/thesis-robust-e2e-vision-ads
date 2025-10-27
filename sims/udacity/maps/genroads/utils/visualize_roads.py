"""
Visualize Udacity roads defined in sims.udacity.configs.roads.
Run in an interactive window or execute directly after editing CONFIG.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# ---- Config ----
MODE = "set" # "set" or "names"
SET_NAME = "all"
NAMES: List[str] = ["straight", "r7"]

# add project root & perturbation-drive to path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PD = ROOT / "external" / "perturbation-drive"
if str(PD) not in sys.path:
    sys.path.insert(0, str(PD))

from perturbationdrive import CustomRoadGenerator
from perturbationdrive.RoadGenerator.Roads.road_visualizer import visualize_road
from sims.udacity.configs.genroads import roads as R


def _generate(gen: CustomRoadGenerator, angles: List[int], segs: List[int], start: Tuple[int,int,int,int]):
    gen.generate(starting_pos=start, angles=angles, seg_length=segs)
    return gen.previous_road

def _viz(name: str, spec: dict) -> None:
    gen = CustomRoadGenerator(num_control_nodes=len(spec["angles"]))
    road = _generate(gen, spec["angles"], spec["segs"], spec.get("start", R.DEFAULT_START))
    visualize_road(road, name)

def _viz_set(set_name: str) -> None:
    for name, spec in R.pick(set_name):
        _viz(name, spec)

def _viz_names(names: Iterable[str]) -> None:
    for n in R.resolve(list(names)):
        _viz(n, R.ROADS[n])

def main() -> int:
    if MODE == "set":
        _viz_set(SET_NAME if isinstance(SET_NAME, str) else R.SELECT)
    elif MODE == "names":
        _viz_names(NAMES)
    else:
        raise ValueError("MODE must be 'set' or 'names'")
    return 0

if __name__ == "__main__":
    main()
