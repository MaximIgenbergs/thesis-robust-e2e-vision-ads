from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from metrics.utils import read_json, try_read_json, safe_str


BASELINE_NAME_DEFAULT = "baseline"


def _infer_model_and_ts(run_dir_name: str) -> Tuple[str, str]:
    # run_id examples: dave2_gru_20251214_203816
    parts = run_dir_name.split("_")
    if len(parts) >= 3:
        model = "_".join(parts[:-2])
        ts = "_".join(parts[-2:])
        return model, ts
    return run_dir_name, "unknown"


def infer_num_segments_from_snapshot(config_snapshot: Dict[str, Any]) -> int:
    # Your snapshot uses configs.segments as a list
    cfg = (config_snapshot or {}).get("configs") or {}
    segs = cfg.get("segments")
    return int(len(segs)) if isinstance(segs, list) else 0


def _normalize_action_list(actions: Any) -> List[List[float]]:
    """
    Normalizes various action formats to: [[steer, throttle], ...]
    Handles GenRoads pid_actions nesting: [[[s,t]], [[s,t]], ...]
    """
    if actions is None:
        return []
    if not isinstance(actions, list):
        return []
    out: List[List[float]] = []
    for a in actions:
        # genroads pid_actions: a might be [[s,t]]
        if isinstance(a, list) and len(a) == 1 and isinstance(a[0], list):
            a = a[0]
        if isinstance(a, list) and len(a) >= 2:
            try:
                out.append([float(a[0]), float(a[1])])
            except Exception:
                continue
    return out


def _coerce_log_entries(log_json: Any) -> List[Dict[str, Any]]:
    """
    Your log.json is a list of dict entries.
    But we accept dict too.
    """
    if isinstance(log_json, list):
        return [x for x in log_json if isinstance(x, dict)]
    if isinstance(log_json, dict):
        return [log_json]
    return []


def iter_udacity_entries(run_dir: Path) -> Iterator[Dict[str, Any]]:
    """
    Yields normalized "entry-level" records:
      - Jungle: one entry per segment within episodes/<id>/log.json list
      - GenRoads: usually one entry, but handles multiple entries if present
    """
    manifest_path = run_dir / "manifest.json"
    manifest = try_read_json(manifest_path)
    if not isinstance(manifest, dict):
        return

    config_snapshot = try_read_json(run_dir / "config_snapshot.json")
    if not isinstance(config_snapshot, dict):
        config_snapshot = {}

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        return

    for ep in episodes:
        if not isinstance(ep, dict):
            continue

        ep_id = safe_str(ep.get("id"), default="unknown")
        log_rel = safe_str(ep.get("log"), default=f"episodes/{ep_id}/log.json")
        log_path = run_dir / log_rel

        meta_path = run_dir / "episodes" / ep_id / "meta.json"
        meta = try_read_json(meta_path)
        if not isinstance(meta, dict):
            meta = {}

        log_json = try_read_json(log_path)
        entries = _coerce_log_entries(log_json)

        yield {
            "run_dir": str(run_dir),
            "manifest": manifest,
            "config_snapshot": config_snapshot,
            "episode_id": ep_id,
            "episode_manifest": ep,   # includes perturbation/severity/road/status
            "meta": meta,
            "log_entries": entries,
        }


def normalize_udacity_entry(
    raw: Dict[str, Any],
    map_name: str,
    test_type: str,
    baseline_name: str = BASELINE_NAME_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Returns a list of normalized entries (one per log entry).
    """
    run_dir = Path(raw["run_dir"])
    manifest: Dict[str, Any] = raw["manifest"]
    config_snapshot: Dict[str, Any] = raw["config_snapshot"]
    meta: Dict[str, Any] = raw["meta"]
    ep_m: Dict[str, Any] = raw["episode_manifest"]
    log_entries: List[Dict[str, Any]] = raw["log_entries"]

    model = safe_str(manifest.get("model"), default=_infer_model_and_ts(run_dir.name)[0])
    run_id = safe_str(manifest.get("run_id"), default=run_dir.name)
    run_ts = safe_str(manifest.get("timestamp"), default=_infer_model_and_ts(run_dir.name)[1])

    # manifest is authoritative for perturbation/severity/road
    pert = ep_m.get("perturbation")
    perturbation = baseline_name if pert is None else safe_str(pert, default=baseline_name)

    try:
        severity = int(ep_m.get("severity", 0))
    except Exception:
        severity = 0

    road = safe_str(ep_m.get("road"), default=map_name)

    # Jungle segments are listed in meta["segs"] (ordered)
    seg_ids = meta.get("segs")
    seg_ids_list = seg_ids if isinstance(seg_ids, list) else []

    # num segments from snapshot (preferred), else from meta["segs"]
    num_segments = infer_num_segments_from_snapshot(config_snapshot)
    if num_segments <= 0 and map_name.lower() == "jungle":
        num_segments = len(seg_ids_list)

    out: List[Dict[str, Any]] = []
    for idx, entry in enumerate(log_entries):
        # signals are lists inside each entry
        xte = entry.get("xte") or []
        angle_err = entry.get("angle_diff") if "angle_diff" in entry else (entry.get("angle_errors") or [])

        actions = _normalize_action_list(entry.get("actions"))
        pid_actions = _normalize_action_list(entry.get("pid_actions"))

        is_success = bool(entry.get("isSuccess", False))
        timeout = bool(entry.get("timeout", False))

        # task_id:
        # - jungle: segment id by entry index (meta["segs"][idx])
        # - genroads: road id
        if map_name.lower() == "jungle":
            seg_id = safe_str(seg_ids_list[idx], default=f"segment_{idx:02d}") if idx < len(seg_ids_list) else f"segment_{idx:02d}"
            task_id = seg_id
        else:
            task_id = road

        out.append({
            "sim": "udacity",
            "map": map_name,
            "test_type": test_type,
            "model": model,
            "run_id": run_id,
            "run_ts": run_ts,
            # unique identifier per entry
            "episode_folder": raw["episode_id"],
            "entry_index": idx,
            "entry_id": f"{raw['episode_id']}_{idx:02d}",
            "task_id": task_id,
            "road": road,
            "perturbation": perturbation,
            "severity": severity,
            "is_success": is_success,
            "timeout": timeout,
            "xte": xte,
            "angle_err": angle_err,
            "actions": actions,
            "pid_actions": pid_actions,
            "num_segments": num_segments,
        })

    return out
