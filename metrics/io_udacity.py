from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from metrics.utils import try_read_json, safe_str, first_present


BASELINE_NAME_DEFAULT = "baseline"


def _infer_model_and_ts(run_dir_name: str) -> Tuple[str, str]:
    # Handles: "<model>_<YYYYmmdd-HHMMSS>" or "<model>_<YYYYmmdd_HHMMSS>"
    parts = run_dir_name.split("_")
    if len(parts) >= 2:
        model = "_".join(parts[:-1])
        ts = parts[-1]
        return model, ts
    return run_dir_name, "unknown"


def _segments_from_config_snapshot(config_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfgs = (config_snapshot or {}).get("configs") or {}
    seg_cfg = cfgs.get("segments")

    # seg_cfg might be:
    # - dict with key "segments": [...]
    # - list of segment dicts
    if isinstance(seg_cfg, dict):
        segs = seg_cfg.get("segments")
        return segs if isinstance(segs, list) else []
    if isinstance(seg_cfg, list):
        return seg_cfg
    return []


def infer_num_segments(config_snapshot: Dict[str, Any]) -> int:
    segs = _segments_from_config_snapshot(config_snapshot)
    return int(len(segs))


def _extract_scenario_fields(log: Dict[str, Any], meta: Dict[str, Any], baseline_name: str) -> Dict[str, Any]:
    scenario = log.get("scenario") or meta.get("scenario") or {}
    if not isinstance(scenario, dict):
        scenario = {}

    perturbation = first_present(
        scenario,
        ["perturbation", "corruption", "name", "perturbation_name", "filter", "effect"],
    )
    if perturbation is None:
        perturbation = first_present(meta, ["perturbation", "corruption", "name", "perturbation_name"])

    perturbation = safe_str(perturbation, default=baseline_name)

    severity = first_present(scenario, ["level", "severity", "s"])
    if severity is None:
        severity = first_present(meta, ["level", "severity", "s"])

    # baseline: enforce severity 0 if missing/invalid
    try:
        severity_i = int(severity) if severity is not None else (0 if perturbation == baseline_name else -1)
    except Exception:
        severity_i = 0 if perturbation == baseline_name else -1

    segment_id = first_present(meta, ["segment_id", "segment", "segmentId", "segment_name"])
    if segment_id is None:
        segment_id = first_present(scenario, ["segment_id", "segment", "segmentId", "id"])
    segment_id = safe_str(segment_id, default="unknown")

    road_id = first_present(meta, ["road_id", "road", "roadId"])
    if road_id is None:
        road_id = first_present(scenario, ["road_id", "road", "roadId"])
    road_id = safe_str(road_id, default="unknown")

    return {
        "perturbation": perturbation,
        "severity": severity_i,
        "segment_id": segment_id,
        "road_id": road_id,
    }


def iter_udacity_episodes(run_dir: Path) -> Iterator[Dict[str, Any]]:
    manifest = try_read_json(run_dir / "manifest.json") or {}
    config_snapshot = try_read_json(run_dir / "config_snapshot.json") or {}

    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        return

    for ep_dir in sorted(episodes_dir.glob("*")):
        if not ep_dir.is_dir():
            continue
        meta = try_read_json(ep_dir / "meta.json") or {}
        log = try_read_json(ep_dir / "log.json") or {}
        if not log:
            continue
        yield {
            "run_dir": str(run_dir),
            "episode_dir": str(ep_dir),
            "manifest": manifest,
            "config_snapshot": config_snapshot,
            "meta": meta,
            "log": log,
        }


def normalize_udacity_episode(
    raw: Dict[str, Any],
    map_name: str,
    test_type: str,
    baseline_name: str = BASELINE_NAME_DEFAULT,
) -> Dict[str, Any]:
    run_dir = Path(raw["run_dir"])
    model, run_ts = _infer_model_and_ts(run_dir.name)

    meta: Dict[str, Any] = raw.get("meta") or {}
    log: Dict[str, Any] = raw.get("log") or {}
    config_snapshot: Dict[str, Any] = raw.get("config_snapshot") or {}

    scenario_fields = _extract_scenario_fields(log, meta, baseline_name)

    is_success = bool(first_present(log, ["isSuccess", "is_success", "success"]) or False)
    timeout = bool(first_present(log, ["timeout", "timed_out"]) or False)

    xte = log.get("xte") or []
    angle = first_present(log, ["angle_diff", "angle_errors"]) or []
    actions = log.get("actions") or []
    pid_actions = log.get("pid_actions") or []

    # Jungle: task = segment_id, GenRoads: task = road_id
    task_id = scenario_fields["segment_id"] if map_name.lower() == "jungle" else scenario_fields["road_id"]

    num_segments = infer_num_segments(config_snapshot) if map_name.lower() == "jungle" else 0

    return {
        "sim": "udacity",
        "map": map_name,
        "test_type": test_type,
        "model": model,
        "run_ts": run_ts,
        "task_id": task_id,
        "episode_id": Path(raw["episode_dir"]).name,
        "perturbation": scenario_fields["perturbation"],
        "severity": scenario_fields["severity"],
        "is_success": is_success,
        "timeout": timeout,
        "xte": xte,
        "angle_err": angle,
        "actions": actions,
        "pid_actions": pid_actions,
        "num_segments": num_segments,
    }
