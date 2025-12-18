from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from metrics.utils import try_read_json, safe_str
from metrics.failure_udacity import (
    has_failure_for_task,
    first_failure_step_for_task,
    first_failure_step_from_pd_steps,
    infer_stop_idx,
)

BASELINE_NAME_DEFAULT = "baseline"


def _infer_model_and_ts(run_dir_name: str) -> Tuple[str, str]:
    parts = run_dir_name.split("_")
    if len(parts) >= 3:
        model = "_".join(parts[:-2])
        ts = "_".join(parts[-2:])
        return model, ts
    return run_dir_name, "unknown"


def infer_num_segments_from_snapshot(config_snapshot: Dict[str, Any]) -> int:
    cfg = (config_snapshot or {}).get("configs") or {}
    segs = cfg.get("segments")
    return int(len(segs)) if isinstance(segs, list) else 0


def _normalize_action_list(actions: Any) -> List[List[float]]:
    if actions is None or not isinstance(actions, list):
        return []
    out: List[List[float]] = []
    for a in actions:
        if isinstance(a, list) and len(a) == 1 and isinstance(a[0], list):
            a = a[0]
        if isinstance(a, list) and len(a) >= 2:
            try:
                out.append([float(a[0]), float(a[1])])
            except Exception:
                continue
    return out


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
        return float(x)
    except Exception:
        return None


def _parse_pd_log(pd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert pd_log.json (steps) into a single "entry" compatible with the rest of the pipeline:
      - xte: list[cte]
      - angle_errors: list[angle]
      - actions: [[steer, throttle], ...] (actual_*, fallback model_*)
      - pid_actions: [] (not present in pd_log)
      - isSuccess / timeout inferred from last step info
    """
    steps = pd.get("steps")
    if not isinstance(steps, list) or not steps:
        return None

    cte_seq: List[float] = []
    ang_seq: List[float] = []
    actions: List[List[float]] = []

    last_info: Dict[str, Any] = {}
    for s in steps:
        if not isinstance(s, dict):
            continue
        info = s.get("info") if isinstance(s.get("info"), dict) else {}
        pid_state = s.get("pid_state") if isinstance(s.get("pid_state"), dict) else {}

        # cte
        cte = _to_float(info.get("cte", None))
        if cte is None:
            cte = _to_float(pid_state.get("cte", None))
        if cte is not None:
            cte_seq.append(cte)

        # angle (string in your sample)
        ang = _to_float(info.get("angle", None))
        if ang is None:
            ang = _to_float(pid_state.get("angle", None))
        if ang is not None:
            ang_seq.append(ang)

        # actions
        steer = _to_float(s.get("actual_steer", None))
        thr = _to_float(s.get("actual_throttle", None))
        if steer is None:
            steer = _to_float(s.get("model_steer", None))
        if thr is None:
            thr = _to_float(s.get("model_throttle", None))
        if steer is not None and thr is not None:
            actions.append([steer, thr])

        last_info = info

    # infer success/timeout from the last step info if present
    is_success_val = last_info.get("is_success", 0)
    is_success = bool(int(is_success_val) == 1) if isinstance(is_success_val, (int, float)) else False
    timeout = bool(last_info.get("timeout", False))

    return {
        # pipeline expects these names
        "xte": cte_seq,
        "angle_errors": ang_seq,
        "actions": actions,
        "pid_actions": [],

        "isSuccess": is_success,
        "timeout": timeout,

        # keep raw steps for failure detection
        "_pd_steps": steps,
    }


def _coerce_log_entries(log_json: Any) -> List[Dict[str, Any]]:
    """
    Jungle: log.json is a list of entry dicts.
    GenRoads: pd_log.json is a dict with {"steps":[...]} -> we convert to one entry dict.
    """
    if isinstance(log_json, list):
        return [x for x in log_json if isinstance(x, dict)]
    if isinstance(log_json, dict):
        if "steps" in log_json:
            one = _parse_pd_log(log_json)
            return [one] if isinstance(one, dict) else []
        return [log_json]
    return []


def iter_udacity_entries(run_dir: Path) -> Iterator[Dict[str, Any]]:
    manifest = try_read_json(run_dir / "manifest.json")
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

        # manifest points to episodes/<id>/log.json, but GenRoads uses episodes/<id>/pd_log.json
        log_rel = safe_str(ep.get("log"), default=f"episodes/{ep_id}/log.json")
        log_path = run_dir / log_rel
        pd_log_path = run_dir / "episodes" / ep_id / "pd_log.json"

        meta_path = run_dir / "episodes" / ep_id / "meta.json"
        events_path = run_dir / "episodes" / ep_id / "events.json"

        meta = try_read_json(meta_path)
        if not isinstance(meta, dict):
            meta = {}

        events = try_read_json(events_path)
        if not isinstance(events, list):
            events = []

        # choose the actual log file
        log_json = None
        if pd_log_path.exists():
            log_json = try_read_json(pd_log_path)
        else:
            log_json = try_read_json(log_path)

        entries = _coerce_log_entries(log_json)

        yield {
            "run_dir": str(run_dir),
            "manifest": manifest,
            "config_snapshot": config_snapshot,
            "episode_id": ep_id,
            "episode_manifest": ep,
            "meta": meta,
            "events": events,
            "log_entries": entries,
        }


def normalize_udacity_entry(
    raw: Dict[str, Any],
    map_name: str,
    test_type: str,
    baseline_name: str = BASELINE_NAME_DEFAULT,
) -> List[Dict[str, Any]]:
    run_dir = Path(raw["run_dir"])
    manifest: Dict[str, Any] = raw["manifest"]
    config_snapshot: Dict[str, Any] = raw["config_snapshot"]
    meta: Dict[str, Any] = raw["meta"]
    events: List[Dict[str, Any]] = raw.get("events") or []
    ep_m: Dict[str, Any] = raw["episode_manifest"]
    log_entries: List[Dict[str, Any]] = raw["log_entries"]

    model = safe_str(manifest.get("model"), default=_infer_model_and_ts(run_dir.name)[0])
    run_id = safe_str(manifest.get("run_id"), default=run_dir.name)
    run_ts = safe_str(manifest.get("timestamp"), default=_infer_model_and_ts(run_dir.name)[1])

    # ---- perturbation normalization (treat "none" / "" / None as baseline) ----
    pert_raw = ep_m.get("perturbation")
    pert_s = "" if pert_raw is None else str(pert_raw).strip()
    if pert_raw is None or pert_s == "" or pert_s.lower() == "none":
        perturbation = baseline_name
        severity = 0
    else:
        perturbation = safe_str(pert_raw, default=baseline_name)
        try:
            severity = int(ep_m.get("severity", 0))
        except Exception:
            severity = 0

    road = safe_str(ep_m.get("road"), default=map_name)

    seg_ids_list = meta.get("segs") if isinstance(meta.get("segs"), list) else []
    num_segments = infer_num_segments_from_snapshot(config_snapshot)
    if num_segments <= 0 and map_name.lower() == "jungle":
        num_segments = len(seg_ids_list)

    out: List[Dict[str, Any]] = []

    for idx, entry in enumerate(log_entries):
        xte = entry.get("xte") or []
        angle_err = entry.get("angle_diff") if "angle_diff" in entry else (entry.get("angle_errors") or [])

        actions = _normalize_action_list(entry.get("actions"))
        pid_actions = _normalize_action_list(entry.get("pid_actions"))

        timeout = bool(entry.get("timeout", False))

        # task_id resolution
        if map_name.lower() == "jungle":
            seg_id = (
                safe_str(seg_ids_list[idx], default=f"segment_{idx:02d}")
                if idx < len(seg_ids_list)
                else f"segment_{idx:02d}"
            )
            task_id = seg_id
        else:
            task_id = road  # genroads: one road/episode

        # -----------------------------
        # Success flag (authoritative)
        # -----------------------------
        if map_name.lower() == "jungle":
            # events.json is authoritative: any collision/out_of_track in this segment => failure
            seg_failed = has_failure_for_task(events=events, task_id=task_id)
            is_success = (not seg_failed) and (not timeout)
        else:
            # genroads: keep the logged completion flag
            is_success = bool(entry.get("isSuccess", False))

        # -----------------------------
        # pre-fail boundary
        # -----------------------------
        series_len = int(
            min(len(xte), len(angle_err)) if (xte and angle_err) else (len(xte) or len(angle_err) or 0)
        )

        if map_name.lower() == "jungle":
            ev_step = first_failure_step_for_task(events=events, task_id=task_id)
        else:
            steps = entry.get("_pd_steps") if isinstance(entry, dict) else None
            ev_step = first_failure_step_from_pd_steps(steps if isinstance(steps, list) else None)

        # NOTE: infer_stop_idx in the updated version expects (series_len, event_step)
        fail_stop_idx = infer_stop_idx(series_len=series_len, event_step=ev_step)

        out.append({
            "sim": "udacity",
            "map": map_name,
            "test_type": test_type,
            "model": model,
            "run_id": run_id,
            "run_ts": run_ts,

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
            "fail_stop_idx": int(fail_stop_idx),

            "track_error_name": "cte" if map_name.lower() != "jungle" else "xte",
        })

    return out
