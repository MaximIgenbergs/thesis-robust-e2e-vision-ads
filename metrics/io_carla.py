from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from metrics.utils import try_read_json, safe_str


def _infer_model_and_ts(run_dir_name: str) -> Tuple[str, str]:
    parts = run_dir_name.split("_")
    if len(parts) >= 3:
        model = "_".join(parts[:-2])
        ts = "_".join(parts[-2:])
        return model, ts
    return run_dir_name, "unknown"


def _is_blocked_from_record(rec: Dict[str, Any]) -> bool:
    # Typical fields seen across leaderboard forks
    for k in ["blocked", "agent_blocked", "agentBlocked", "blocked_test", "blockedTest"]:
        if k in rec and isinstance(rec[k], bool):
            return rec[k]

    status = str(rec.get("status", "") or rec.get("result", "") or "").lower()
    if "blocked" in status:
        return True
    if "agent blocked" in status:
        return True
    return False


def _driving_score_from_record(rec: Dict[str, Any]) -> Optional[float]:
    # Try common leaderboard keys
    candidates = [
        "driving_score",
        "score_composed",
        "score",
        "drivingScore",
        "scoreComposed",
        "score_total",
    ]
    for k in candidates:
        if k in rec:
            try:
                return float(rec[k])
            except Exception:
                pass

    # Sometimes nested
    scores = rec.get("scores") or rec.get("score") or {}
    if isinstance(scores, dict):
        for k in ["driving_score", "score_composed", "score_total", "composed"]:
            if k in scores:
                try:
                    return float(scores[k])
                except Exception:
                    pass

    return None


def iter_carla_route_records(run_dir: Path) -> Iterator[Dict[str, Any]]:
    """
    Best guess: scan JSON files that contain route records.
    You will likely tweak this once you’re at the faculty machine.
    """
    json_files = list(run_dir.rglob("*.json"))
    for p in json_files:
        data = try_read_json(p)
        if not data:
            continue

        # Pattern A: {"records":[...]}
        if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
            for rec in data["records"]:
                if isinstance(rec, dict):
                    yield {"source": str(p), "record": rec}
            continue

        # Pattern B: {"routes":[...]} or {"results":[...]}
        for key in ["routes", "results"]:
            if isinstance(data, dict) and key in data and isinstance(data[key], list):
                for rec in data[key]:
                    if isinstance(rec, dict):
                        yield {"source": str(p), "record": rec}
                break


def normalize_carla_route_record(
    run_dir: Path,
    test_type: str,
    raw: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    model, run_ts = _infer_model_and_ts(run_dir.name)

    rec: Dict[str, Any] = raw["record"]
    ds = _driving_score_from_record(rec)
    if ds is None:
        return None

    town = safe_str(rec.get("town") or rec.get("map") or rec.get("Town") or "unknown")
    route_id = safe_str(rec.get("route_id") or rec.get("route") or rec.get("id") or "unknown")
    blocked = _is_blocked_from_record(rec)

    return {
        "sim": "carla",
        "map": town,            # town-level
        "test_type": test_type,
        "model": model,
        "run_ts": run_ts,
        "route_id": route_id,
        "driving_score": float(ds),
        "blocked": bool(blocked),
        "source": raw.get("source", ""),
    }
