from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

N_WP = 335
MAX_WP = N_WP - 1  # 334
DEFAULT_MARGIN = 3


@dataclass(frozen=True)
class Template:
    code: str
    control: str            # "steering" | "throttle"
    mode: str               # "add" | "mul" | "set"
    levels: List[Tuple[int, float]]  # (level, factor)
    duration_wp: int
    k_starts: int
    signed: bool = False    # alternate sign for "add" steering templates
    tail_buffer_wp: int = 0  # keep this many waypoints free AFTER end_wp


def _linspace_int(a: int, b: int, k: int) -> List[int]:
    if k <= 1:
        return [int(round((a + b) / 2))]
    if a > b:
        return []
    span = b - a
    out: List[int] = []
    for i in range(k):
        x = a + (span * i) / (k - 1)
        out.append(int(round(x)))
    # de-dup while preserving order
    seen = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    # if rounding collapsed points, fill deterministically
    cur = a
    while len(uniq) < k and cur <= b:
        if cur not in seen:
            seen.add(cur)
            uniq.append(cur)
        cur += 1
    return uniq[:k]


def _find_roads_yaml() -> Path:
    rel = Path("scripts/udacity/maps/genroads/roads.yaml")
    candidates = [Path.cwd() / rel]
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents)[:6]:
        candidates.append(p / rel)
    for c in candidates:
        if c.exists():
            return c
    return Path(rel)


def _make_starts(k: int, duration_wp: int, margin: int, tail_buffer_wp: int = 0) -> List[int]:
    # inclusive interval => end = start + duration - 1
    start_min = margin
    start_max = MAX_WP - margin - (duration_wp - 1) - tail_buffer_wp
    if start_max < start_min:
        raise ValueError(
            f"Invalid start range: start_min={start_min}, start_max={start_max} "
            f"(duration_wp={duration_wp}, margin={margin}, tail_buffer_wp={tail_buffer_wp}, MAX_WP={MAX_WP})"
        )
    return _linspace_int(start_min, start_max, k)


def _scenario_name(code: str, idx: int) -> str:
    return f"{code}{idx:02d}"


def _build_generated_for_road(
    road_name: str,
    templates: List[Template],
    margin: int,
) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    # Baseline (no perturbation)
    scenarios.append(
        {
            "name": "baseline",
            "active": True,
            "road": road_name,
            "level": 0,
            "comment": "",
            "perturbations": [],
        }
    )

    for t in templates:
        starts = _make_starts(t.k_starts, t.duration_wp, margin, t.tail_buffer_wp)

        local_idx = 1
        for (level, base_factor) in t.levels:
            for j, s in enumerate(starts):
                factor = base_factor
                if t.signed and factor != 0.0:
                    sign = -1.0 if (j % 2 == 1) else 1.0
                    factor = float(sign * base_factor)

                e = int(s + t.duration_wp - 1)  # inclusive end (matches runner)

                scen = {
                    "name": _scenario_name(t.code, local_idx),
                    "active": True,
                    "road": road_name,
                    "level": int(level),
                    "comment": f"{t.code}: {t.control} {t.mode} factor={factor} start={s} end={e}",
                    "perturbations": [
                        {
                            "control": t.control,
                            "mode": t.mode,
                            "factor": float(factor),
                            "start_wp": int(s),
                            "end_wp": int(e),
                            "comment": "",
                        }
                    ],
                }
                scenarios.append(scen)
                local_idx += 1

    return scenarios


def generate(
    roads_yaml: Optional[Path],
    out_path: Path,
    margin: int,
) -> Dict[str, List[Dict[str, Any]]]:
    _ = roads_yaml or _find_roads_yaml()  # kept for compatibility / future use

    road_names = [
        "wide_hairpin_r", "wide_hairpin_l",
        "hairpin_r", "hairpin_l",
        "wide_chicane_lr", "wide_chicane_rl",
        "chicane_lr", "chicane_rl",
    ]

    # Topology-agnostic placement via stratified start points.
    # tail_buffer_wp prevents "impulse" perturbations from starting too close to the end,
    # so there is always room for a reaction phase after the perturbation ends.
    templates: List[Template] = [
        Template(
            code="spulse",
            control="steering",
            mode="add",
            levels=[(1, 0.25), (2, 0.40), (3, 0.60)],
            duration_wp=8,
            k_starts=6,
            signed=True,
            tail_buffer_wp=60,
        ),
        Template(
            code="sfreeze",
            control="steering",
            mode="set",
            levels=[(1, 0.0)],
            duration_wp=10,
            k_starts=6,
            tail_buffer_wp=60,
        ),
        Template(
            code="sgain",
            control="steering",
            mode="mul",
            levels=[(1, 1.40)],
            duration_wp=40,
            k_starts=6,
            tail_buffer_wp=60,
        ),
        Template(
            code="sbias",
            control="steering",
            mode="add",
            levels=[(1, 0.20)],
            duration_wp=40,
            k_starts=6,
            signed=True,
            tail_buffer_wp=60,
        ),
        Template(
            code="tboost",
            control="throttle",
            mode="mul",
            levels=[(1, 1.20)],
            duration_wp=100,
            k_starts=2,
            tail_buffer_wp=60,
        ),
        Template( # friendly, makes it easier for the model
            code="btap",
            control="throttle",
            mode="set",
            levels=[(1, -1.0)],
            duration_wp=5,
            k_starts=5,
            tail_buffer_wp=60,
        ),
    ]


    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in road_names:
        out[r] = _build_generated_for_road(r, templates, margin)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    total = sum(len(v) for v in out.values())
    print(f"[scripts:genroads:generate_scenarios] roads={len(out)} total_scenarios={total} -> {out_path}")
    per = ", ".join(f"{k}:{len(v)}" for k, v in out.items())
    print(f"[scripts:genroads:generate_scenarios] per-road counts: {per}")
    print("[scripts:genroads:generate_scenarios] NOTE: end_wp is inclusive (matches runner).")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roads-yaml",
        type=str,
        default=None,
        help="Path to scripts/udacity/maps/genroads/roads.yaml (optional).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="generated_scenarios.yaml",
        help="Output YAML file path.",
    )
    ap.add_argument(
        "--margin",
        type=int,
        default=DEFAULT_MARGIN,
        help="Waypoint margin to avoid near-start/end placements.",
    )
    args = ap.parse_args()

    roads_yaml = Path(args.roads_yaml) if args.roads_yaml else None
    out_path = Path(args.out)

    generate(roads_yaml=roads_yaml, out_path=out_path, margin=int(args.margin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
