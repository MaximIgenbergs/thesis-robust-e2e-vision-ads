"""
Road definitions for Udacity road generation.

- ROADS: dict of named roads -> {angles, segs}
- SETS:  dict of named presets (lists of road names)
- SELECT: which roads to use by default (either a set name or a list of names)
"""

from __future__ import annotations
from typing import Dict, List

ROADS: Dict[str, Dict[str, List[int]]] = {
    # Road 1: straight
    "straight": {"angles": [0, 0, 0, 0, 0, 0, 0, 0], "segs": [25, 25, 25, 25, 25, 25, 25, 25]},

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

    # Custom roads (designed to be as challenging as possible)
    "a": {"angles": [0, 0, -35, 28, -24, 20, 30, 35], "segs": [45,35,15,16,14,22,18,22]},
    "b": {"angles": [0, 0, 35, -28, 24, -20, -30, -35], "segs": [45,35,15,16,14,22,18,22]},
    "c": {"angles": [0, 0, -35, 35, -35, 20, 30, 35], "segs": [45,45,15,15,15,22,18,22]},

    # Data-collection roads
    # Randomly generated roads
    "random_000": {"angles": [-4, 16, -6, 14, 17, -10, 0, -19, 5, 9], "segs": [22, 24, 29, 23, 30, 28, 26, 25, 29, 22]},
    "random_001": {"angles": [-42, 31, -9, -12, 28, -45, 11, 38, -42, 36], "segs": [27, 24, 23, 22, 23, 29, 22, 24, 23, 25]},
    "random_002": {"angles": [2, -13, -13, -13, 1, 14, 9, -4, -3, -13], "segs": [28, 20, 27, 21, 24, 21, 22, 29, 22, 29]},
    "random_003": {"angles": [-40, 3, -6, 31, 0, -26, 10, -5, -7, 17], "segs": [29, 22, 30, 28, 23, 25, 22, 22, 23, 27]},
    "random_004": {"angles": [6, 10, -15, 3, 4, -5, -5, -1, -15, 5], "segs": [27, 27, 21, 25, 29, 24, 25, 30, 28, 30]},
    "random_005": {"angles": [1, 35, 28, 13, -37, -3, -1, 25, -30, -4], "segs": [22, 20, 24, 24, 23, 30, 24, 28, 20, 20]},
    "random_006": {"angles": [9, -13, -3, -2, 2, 7, -3, -4, -12, 5], "segs": [26, 24, 25, 24, 28, 21, 30, 22, 26, 22]},
    "random_007": {"angles": [-28, -8, -28, 31, -4, -4, 35, -29, -33, -5], "segs": [29, 22, 25, 25, 30, 27, 24, 29, 20, 24]},
    "random_008": {"angles": [3, 9, -7, 11, 8, -3, 2, 12, 12, 4], "segs": [29, 28, 30, 29, 23, 20, 21, 21, 24, 24]},
    "random_009": {"angles": [14, -3, -30, -5, 29, -2, -15, 26, -26, 18], "segs": [20, 26, 26, 20, 22, 20, 25, 28, 22, 21]},
    "random_010": {"angles": [-9, -7, -8, 2, -3, 4, 6, 0, -2, -6], "segs": [22, 28, 22, 30, 22, 20, 22, 29, 28, 28]},
    "random_011": {"angles": [1, 5, -19, -10, -4, 25, 0, 22, 5, -13], "segs": [26, 30, 22, 25, 28, 29, 26, 26, 23, 30]},
    "random_012": {"angles": [-2, -4, -4, 1, 3, 0, 2, -7, -1, -3], "segs": [21, 21, 24, 22, 20, 22, 20, 22, 28, 30]},
    "random_013": {"angles": [14, 5, 20, 1, 7, -7, -9, -2, 2, -10], "segs": [27, 21, 22, 25, 25, 20, 29, 24, 23, 29]},
    "random_014": {"angles": [-4, 4, -5, 2, -5, -2, -3, -4, -5, 2], "segs": [30, 24, 20, 22, 26, 21, 25, 22, 26, 23]},
    "random_015": {"angles": [-8, -13, -14, 1, -13, 7, 5, 3, -2, -1], "segs": [24, 28, 22, 26, 30, 23, 27, 20, 20, 27]},
    "random_016": {"angles": [0, -2, 1, 1, 3, -1, 3, -3, -3, 0], "segs": [28, 20, 25, 26, 28, 26, 22, 23, 20, 30]},
    "random_017": {"angles": [-4, -2, 10, -8, 8, 5, -8, -7, 1, 9], "segs": [24, 27, 20, 29, 28, 26, 28, 20, 25, 22]},
    
    # Circles
    "simple_018": {"angles": [50, 50, 50, 50, 50, 50, 50, 50], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_019": {"angles": [-50, -50, -50, -50, -50, -50, -50, -50], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_020": {"angles": [45, 45, 45, 45, 45, 45, 45, 45], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_021": {"angles": [-45, -45, -45, -45, -45, -45, -45, -45], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_022": {"angles": [40, 40, 40, 40, 40, 40, 40, 40], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_023": {"angles": [-40, -40, -40, -40, -40, -40, -40, -40], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_024": {"angles": [35, 35, 35, 35, 35, 35, 35, 35], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_025": {"angles": [-35, -35, -35, -35, -35, -35, -35, -35], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_026": {"angles": [30, 30, 30, 30, 30, 30, 30, 30], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_027": {"angles": [-30, -30, -30, -30, -30, -30, -30, -30], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_028": {"angles": [25, 25, 25, 25, 25, 25, 25, 25], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_029": {"angles": [-25, -25, -25, -25, -25, -25, -25, -25], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_030": {"angles": [20, 20, 20, 20, 20, 20, 20, 20], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_031": {"angles": [-20, -20, -20, -20, -20, -20, -20, -20], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_032": {"angles": [15, 15, 15, 15, 15, 15, 15, 15], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_033": {"angles": [-15, -15, -15, -15, -15, -15, -15, -15], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_034": {"angles": [10, 10, 10, 10, 10, 10, 10, 10], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_035": {"angles": [-10, -10, -10, -10, -10, -10, -10, -10], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_036": {"angles": [5, 5, 5, 5, 5, 5, 5, 5], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    "simple_037": {"angles": [-5, -5, -5, -5, -5, -5, -5, -5], "segs": [10, 10, 10, 10, 10, 10, 10, 10]},
    
    # Alternating Curves
    "swiggly_038": {"angles": [50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50], "segs": [20]*20},
    "swiggly_039": {"angles": [-50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50, -50, 50], "segs": [20]*20},
    "swiggly_040": {"angles": [45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45], "segs": [20]*20},
    "swiggly_041": {"angles": [-45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45, -45, 45], "segs": [20]*20},
    "swiggly_042": {"angles": [40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40], "segs": [20]*20},
    "swiggly_043": {"angles": [-40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40, -40, 40], "segs": [20]*20},
    "swiggly_044": {"angles": [35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35], "segs": [20]*20},
    "swiggly_045": {"angles": [-35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35, -35, 35], "segs": [20]*20},
    "swiggly_046": {"angles": [30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30], "segs": [20]*20},
    "swiggly_047": {"angles": [-30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30, -30, 30], "segs": [20]*20},
    "swiggly_048": {"angles": [25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25], "segs": [20]*20},
    "swiggly_049": {"angles": [-25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25, -25, 25], "segs": [20]*20},
    "swiggly_050": {"angles": [20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20], "segs": [20]*20},
    "swiggly_051": {"angles": [-20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20], "segs": [20]*20},
    "swiggly_052": {"angles": [15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15], "segs": [20]*20},
    "swiggly_053": {"angles": [-15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15, -15, 15], "segs": [20]*20},
    "swiggly_054": {"angles": [10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10], "segs": [20]*20},
    "swiggly_055": {"angles": [-10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10], "segs": [20]*20},
    "swiggly_056": {"angles": [5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5], "segs": [20]*20},
    "swiggly_057": {"angles": [-5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5], "segs": [20]*20},
}

# Named presets (road sets). Values are lists of keys from ROADS.
SETS: Dict[str, List[str]] = {
    "baseline10": ["straight", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"],
    "trio": ["a", "b", "c"],
    "all": list(ROADS.keys()),
    "data_collection": [
        "random_000", "random_001", "random_002", "random_003", "random_004", "random_005", "random_006", "random_007", "random_008", "random_009", "random_010", "random_011", "random_012", "random_013", "random_014", "random_015", "random_016", "random_017",
        "simple_018", "simple_019", "simple_020", "simple_021", "simple_022", "simple_023", "simple_024", "simple_025", "simple_026", "simple_027", "simple_028", "simple_029", "simple_030", "simple_031", "simple_032", "simple_033", "simple_034", "simple_035", "simple_036", "simple_037",
        "swiggly_038", "swiggly_039", "swiggly_040", "swiggly_041", "swiggly_042", "swiggly_043", "swiggly_044", "swiggly_045", "swiggly_046", "swiggly_047", "swiggly_048", "swiggly_049", "swiggly_050", "swiggly_051", "swiggly_052", "swiggly_053", "swiggly_054", "swiggly_055", "swiggly_056", "swiggly_057"
    ],
}

# Default selection (unchanged)
SELECT: List[str] | str = ["c"] # example: "trio" or ["straight"]

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
