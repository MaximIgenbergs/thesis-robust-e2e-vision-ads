"""
Visualize Udacity roads defined in ROADS_PATH.
Edit SET_NAME below to choose which set from roads.yaml to visualize.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml

from scripts import abs_path
from scripts.udacity.maps.genroads.utils.load_roads import load_roads
from perturbationdrive import CustomRoadGenerator
from perturbationdrive.RoadGenerator.Roads.road_visualizer import visualize_road

SET_NAME = "generalization"

ROADS_PATH = abs_path("scripts/udacity/maps/genroads/roads/roads.yaml")

# Default starting pose used for visualization (x, y, yaw_deg, speed)
# Exact numbers don't matter for visualization; adjust if you care about absolute position.
DEFAULT_START: Tuple[int, int, int, int] = (0, 0, 0, 10)


def generate(gen: CustomRoadGenerator, angles: List[int], segs: List[int], start: Tuple[int, int, int, int]):
    gen.generate(starting_pos=start, angles=angles, seg_length=segs)
    return gen.previous_road


def visualize(name: str, spec: Dict[str, Any]) -> None:
    angles = spec["angles"]
    segs = spec["segs"]
    start = DEFAULT_START

    gen = CustomRoadGenerator(num_control_nodes=len(angles))
    road = generate(gen, angles, segs, start)
    visualize_road(road, name)


def visualize_set(set_name: str, roads_def: Dict[str, Dict[str, Any]], sets_def: Dict[str, List[str]]) -> None:
    if set_name not in sets_def:
        raise KeyError(f"Unknown road set '{set_name}'. Known sets: {list(sets_def.keys())}")

    for road_name in sets_def[set_name]:
        if road_name not in roads_def:
            raise KeyError(f"Road '{road_name}' referenced in set '{set_name}' but not defined in the 'roads' section of {ROADS_PATH}")
        visualize(road_name, roads_def[road_name])


def main() -> int:
    roads_def, sets_def = load_roads(ROADS_PATH)
    visualize_set(SET_NAME, roads_def, sets_def)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
