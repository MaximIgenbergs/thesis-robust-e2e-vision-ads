"""
Visualize Udacity roads defined in ROADS_PATH.
Edit SET_NAME below to choose which set from roads.yaml to visualize.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml

from scripts import abs_path
from perturbationdrive import CustomRoadGenerator
from perturbationdrive.RoadGenerator.Roads.road_visualizer import visualize_road

SET_NAME = "all"

ROADS_PATH = abs_path("scripts/udacity/maps/genroads/configs/roads.yaml")

# Default starting pose used for visualization (x, y, yaw_deg, speed)
# Exact numbers don't matter for visualization; adjust if you care about absolute position.
DEFAULT_START: Tuple[int, int, int, int] = (0, 0, 0, 10)


def load_roads(yaml_path: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"roads.yaml not found at: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    roads_def = cfg.get("roads") or {}
    sets_def = cfg.get("sets") or {}

    if not roads_def:
        raise ValueError(f"No 'roads' section in {yaml_path}")
    if not sets_def:
        raise ValueError(f"No 'sets' section in {yaml_path}")

    # Basic validation
    for name, spec in roads_def.items():
        angles = spec.get("angles")
        segs = spec.get("segs")
        if not isinstance(angles, list) or not isinstance(segs, list):
            raise ValueError(f"[road:{name}] 'angles' and 'segs' must be lists (file: {yaml_path})")
        if len(angles) != len(segs):
            raise ValueError(f"[road:{name}] length mismatch: angles({len(angles)}) != segs({len(segs)}) (file: {yaml_path})")

    return roads_def, sets_def


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
