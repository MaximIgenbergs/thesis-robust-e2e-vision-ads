from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from metrics.utils import try_read_json, safe_str

# robustness folder: <perturbation>_s<severity>
_CORR_RE = re.compile(r"^(?P<name>.+)_s(?P<sev>\d+)$", re.IGNORECASE)

# filenames we accept
_ALLOWED_PREFIXES = ("simulation_results",)
_ALLOWED_SUFFIXES = (".json",)

IGNORE_LAV = True  # you said: ignore LAV for now


def _looks_like_results_file(p: Path) -> bool:
    n = p.name.lower()
    return n.startswith(_ALLOWED_PREFIXES) and n.endswith(_ALLOWED_SUFFIXES)


def _parse_corr_from_folder(folder_name: str) -> Optional[Tuple[str, int]]:
    m = _CORR_RE.match(folder_name)
    if not m:
        return None
    return m.group("name"), int(m.group("sev"))


def _is_lav_path(p: Path) -> bool:
    if "lav" in p.name.lower():
        return True
    return any(part.lower() == "lav" for part in p.parts)


def iter_carla_result_files(run_root: Path) -> Iterator[Dict[str, Any]]:
    """
    Yields dicts with:
      - kind: "robustness" or "generalization"
      - condition: town group (e.g., Town01_03, Town04) OR corruption folder
      - perturbation, severity (for generalization: baseline, 0)
      - path, data
    """
    for p in sorted(run_root.rglob("*.json")):
        if not _looks_like_results_file(p):
            continue
        if IGNORE_LAV and _is_lav_path(p):
            continue

        data = try_read_json(p)
        if not isinstance(data, dict):
            continue

        rel = p.relative_to(run_root)
        parts = list(rel.parts)

        # Robustness case: .../<pert>_s<sev>/simulation_results.json
        if len(parts) >= 2:
            corr = _parse_corr_from_folder(parts[0])
            if corr is not None:
                pert, sev = corr
                yield {
                    "kind": "robustness",
                    "condition": parts[0],
                    "perturbation": pert,
                    "severity": int(sev),
                    "path": str(p),
                    "data": data,
                }
                continue

        # Generalization case: .../<TownXX...>/.../simulation_results*.json
        condition = parts[0] if parts else "unknown_condition"
        yield {
            "kind": "generalization",
            "condition": condition,
            "perturbation": "baseline",
            "severity": 0,
            "path": str(p),
            "data": data,
        }


def parse_carla_routes(sim_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    cp = sim_results.get("_checkpoint") or {}
    records = cp.get("records")
    if not isinstance(records, list):
        return []
    return [r for r in records if isinstance(r, dict)]


def is_blocked_route(rec: Dict[str, Any]) -> bool:
    status = safe_str(rec.get("status"), default="").lower()
    if "got blocked" in status:
        return True
    infra = rec.get("infractions") or {}
    vb = infra.get("vehicle_blocked")
    if isinstance(vb, list) and len(vb) > 0:
        return True
    return False


def route_driving_score(rec: Dict[str, Any]) -> Optional[float]:
    scores = rec.get("scores") or {}
    try:
        return float(scores.get("score_composed"))
    except Exception:
        return None


def route_penalty(rec: Dict[str, Any]) -> Optional[float]:
    scores = rec.get("scores") or {}
    try:
        return float(scores.get("score_penalty"))
    except Exception:
        return None


def route_completion(rec: Dict[str, Any]) -> Optional[float]:
    scores = rec.get("scores") or {}
    try:
        return float(scores.get("score_route"))
    except Exception:
        return None


def route_duration_game(rec: Dict[str, Any]) -> Optional[float]:
    meta = rec.get("meta") or {}
    try:
        return float(meta.get("duration_game"))
    except Exception:
        return None


def normalize_carla_run(run_root: Path, test_type: str) -> List[Dict[str, Any]]:
    """
    run_root:
      .../runs/carla/<test_type>/<model>/<run_ts>

    Emits route-level rows.
    """
    model = safe_str(run_root.parent.name, default="unknown_model")
    run_ts = safe_str(run_root.name, default="unknown_run")

    rows: List[Dict[str, Any]] = []

    for item in iter_carla_result_files(run_root):
        condition = safe_str(item["condition"], default="unknown_condition")
        perturbation = safe_str(item["perturbation"], default="baseline")
        severity = int(item["severity"])
        source = item["path"]

        routes = parse_carla_routes(item["data"])
        for rec in routes:
            ds = route_driving_score(rec)
            if ds is None:
                continue
            blocked = is_blocked_route(rec)

            rows.append({
                "sim": "carla",
                "map": "multi-town",
                "test_type": test_type,
                "model": model,
                "run_ts": run_ts,
                "condition": condition,          # Town01_03 / Town04 / ... OR perturbation_sX
                "perturbation": perturbation,    # baseline for generalization
                "severity": severity,            # 0 for generalization
                "route_id": safe_str(rec.get("route_id"), default="unknown"),
                "driving_score": float(ds),
                "score_penalty": route_penalty(rec),
                "score_route": route_completion(rec),
                "blocked": int(blocked),
                "duration_game_s": route_duration_game(rec),
                "status": safe_str(rec.get("status"), default=""),
                "source": source,
            })

    return rows
