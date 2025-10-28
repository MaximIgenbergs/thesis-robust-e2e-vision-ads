"""
Road definitions for Udacity road generation.

- ROADS: dict of named roads -> {angles, segs}
- SETS:  dict of named presets (lists of road names)
- SELECT: which roads to use by default (either a set name or a list of names)

Usage:
    from sims.udacity.configs import roads
    for name, spec in roads.pick():
        angles = spec["angles"]; segs = spec["segs"]
        road_gen.generate(starting_pos=bench.simulator.initial_pos, angles=angles, seg_length=segs)
"""

from __future__ import annotations
from typing import Dict, List

ROADS: Dict[str, Dict[str, List[int]]] = {
    # Road 1: straight
    "straight": {
        "angles": [0, 0, 0, 0, 0, 0, 0, 0],
        "segs": [25, 25, 25, 25, 25, 25, 25, 25],
    },

    # Road 2–10: various smoothness/turns
    "r2": {"angles": [0, -20, -10, -11, -13, -25, -24, 4], "segs": [25]*8},
    "r3": {"angles": [0, 24, 24, -5, -7, -24, -20, 17], "segs": [25]*8},
    "r4": {"angles": [0, 6, 8, 20, -3, 20, -13, -27], "segs": [25]*8},
    "r5": {"angles": [0, -2, -30, -8, 2, 2, 4, -30], "segs": [25]*8},
    "r6": {"angles": [0, 0, 27, 35, -35, -35, -6, -35], "segs": [25]*8},
    "r7": {"angles": [0, -27, -20, 35, 23, -3, 4, -35], "segs": [25]*8},
    "r8": {"angles": [0, 30, 30, -30, 32, -32, 25, -12], "segs": [25]*8},
    "r9": {"angles": [0, -35, -20, -35, 35, 35, 35, -35], "segs": [25]*8},
    "r10": {"angles": [0, -35, 0, -17, -35, 35, 6, -22], "segs": [25]*8},
    "a": {"angles": [0, 0, -35, 28, -24, 20, 30, 35], "segs": [45,35,15,16,14,22,18,22]},
    "b": {"angles": [0, 0, 35, -28, 24, -20, -30, -35], "segs": [45,35,15,16,14,22,18,22]},
    "c": {"angles": [0, 0, -35, 35, -35, 20, 30, 35], "segs": [45,45,15,15,15,22,18,22]},
}

# Named presets (road sets). Values are lists of keys from ROADS.
SETS: Dict[str, List[str]] = {
    "baseline10": ["straight", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"],
    "trio": ["a", "b", "c"],
    "all": list(ROADS.keys()),
}

# What to use by default in runners; may be a set name or a list of names.
SELECT: List[str] | str = ["c"]

# ---- Helpers ----

def list_names() -> List[str]:
    return list(ROADS.keys())

def resolve(names_or_set: List[str] | str) -> List[str]:
    if isinstance(names_or_set, str):
        if names_or_set not in SETS:
            raise KeyError(f"Unknown set '{names_or_set}'. Known sets: {list(SETS.keys())}")
        return list(SETS[names_or_set])
    unknown = [n for n in names_or_set if n not in ROADS]
    if unknown:
        raise KeyError(f"Unknown road names: {unknown}. Known: {list(ROADS.keys())}")
    return list(names_or_set)

def pick(names_or_set: List[str] | str = SELECT):
    """
    Yields (name, spec) pairs for the requested selection.
    spec has keys: 'angles', 'segs'.
    """
    for name in resolve(names_or_set):
        spec = ROADS[name]
        _validate(name, spec)
        yield name, spec

def _validate(name: str, spec: Dict) -> None:
    angles = spec.get("angles"); segs = spec.get("segs")
    if not isinstance(angles, list) or not isinstance(segs, list):
        raise ValueError(f"[{name}] 'angles' and 'segs' must be lists.")
    if len(angles) != len(segs):
        raise ValueError(f"[{name}] length mismatch: angles({len(angles)}) != segs({len(segs)})")
