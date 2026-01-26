from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from metrics.utils import try_read_json, safe_str
from metrics.constants import BASELINE_NAME_DEFAULT
from metrics.failure_udacity import (
    has_failure_for_task,
    first_failure_step_for_task,
    first_failure_step_from_pd_steps,
    infer_stop_idx,
)


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


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
        return float(x)
    except Exception:
        return None


def _first_float(*cands: Any) -> Optional[float]:
    for c in cands:
        v = _to_float(c)
        if v is not None:
            return v
    return None


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return []


def _as_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, (int, float)):
        v = _to_float(x)
        return [v] if v is not None else []
    if isinstance(x, list):
        out: List[float] = []
        for a in x:
            v = _to_float(a)
            if v is not None:
                out.append(v)
        return out
    return []


def _normalize_action_pair(a: Any) -> Optional[List[float]]:
    if a is None:
        return None

    if isinstance(a, (list, tuple)):
        if len(a) == 1 and isinstance(a[0], (list, tuple)):
            a = a[0]
        if len(a) >= 2:
            try:
                return [float(a[0]), float(a[1])]
            except Exception:
                return None
        return None

    if isinstance(a, dict):
        steer = _first_float(a.get("steer"), a.get("steering"), a.get("angle"))
        thr = _first_float(a.get("throttle"), a.get("thr"))
        if steer is not None and thr is not None:
            return [float(steer), float(thr)]
        return None

    return None


def _zip_steer_throttle(steer_seq: Any, thr_seq: Any) -> List[List[float]]:
    s = _as_list(steer_seq)
    t = _as_list(thr_seq)
    if not s or not t:
        return []
    out: List[List[float]] = []
    for i in range(min(len(s), len(t))):
        sv = _to_float(s[i])
        tv = _to_float(t[i])
        if sv is None or tv is None:
            continue
        out.append([float(sv), float(tv)])
    return out


def _normalize_action_list(actions: Any) -> List[List[float]]:
    if actions is None:
        return []

    # If already a dict, try common (steer_seq, throttle_seq) patterns or recurse into nested "actions"
    if isinstance(actions, dict):
        if "actions" in actions and actions["actions"] is not None:
            return _normalize_action_list(actions["actions"])

        steer_seq = None
        thr_seq = None

        for k in ("steer", "steering", "angle", "model_steer", "actual_steer"):
            if k in actions:
                steer_seq = actions.get(k)
                break
        for k in ("throttle", "thr", "model_throttle", "actual_throttle"):
            if k in actions:
                thr_seq = actions.get(k)
                break

        zipped = _zip_steer_throttle(steer_seq, thr_seq)
        if zipped:
            return zipped

        pair = _normalize_action_pair(actions)
        return [pair] if pair is not None else []

    # If list, handle:
    # - list-of-pairs
    # - a single pair
    # - [steer_seq, throttle_seq]
    if isinstance(actions, list):
        if len(actions) == 1:
            pair = _normalize_action_pair(actions[0])
            return [pair] if pair is not None else []

        # Special: [steer_seq, throttle_seq]
        if len(actions) == 2 and isinstance(actions[0], list) and isinstance(actions[1], list):
            # if these look like sequences of numbers (not list-of-pairs)
            if actions[0] and not isinstance(actions[0][0], (list, dict)) and actions[1] and not isinstance(actions[1][0], (list, dict)):
                zipped = _zip_steer_throttle(actions[0], actions[1])
                if zipped:
                    return zipped

        out: List[List[float]] = []
        for a in actions:
            pair = _normalize_action_pair(a)
            if pair is not None:
                out.append(pair)
        return out

    # Anything else: try to interpret as one pair
    pair = _normalize_action_pair(actions)
    return [pair] if pair is not None else []


def _extract_pair_from_step(step: Dict[str, Any], keys: List[str]) -> Optional[List[float]]:
    for k in keys:
        if k in step:
            pair = _normalize_action_pair(step.get(k))
            if pair is not None:
                return pair
    info = step.get("info") if isinstance(step.get("info"), dict) else {}
    for k in keys:
        if k in info:
            pair = _normalize_action_pair(info.get(k))
            if pair is not None:
                return pair
    return None


def _parse_pd_log(pd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    steps = pd.get("steps")
    if not isinstance(steps, list) or not steps:
        return None

    cte_seq: List[float] = []
    ang_seq: List[float] = []
    actions: List[List[float]] = []
    pid_actions: List[List[float]] = []

    last_info: Dict[str, Any] = {}
    for s in steps:
        if not isinstance(s, dict):
            continue
        info = s.get("info") if isinstance(s.get("info"), dict) else {}
        pid_state = s.get("pid_state") if isinstance(s.get("pid_state"), dict) else {}

        cte = _first_float(
            info.get("xte"), info.get("cte"),
            pid_state.get("xte"), pid_state.get("cte"),
            s.get("xte"), s.get("cte"),
        )
        if cte is not None:
            cte_seq.append(cte)

        ang = _first_float(
            info.get("angle_diff"), info.get("angle_error"), info.get("angle_err"), info.get("angle_errors"),
            pid_state.get("angle_diff"), pid_state.get("angle_error"), pid_state.get("angle_err"), pid_state.get("angle_errors"),
            s.get("angle_diff"), s.get("angle_error"), s.get("angle_err"), s.get("angle_errors"),
            info.get("angle"), pid_state.get("angle"), s.get("angle"),
        )
        if ang is not None:
            ang_seq.append(ang)

        model_pair = _extract_pair_from_step(s, ["action", "actions", "model_action", "model_actions", "act"])
        if model_pair is None:
            steer = _first_float(s.get("actual_steer"), s.get("model_steer"), info.get("steer"), info.get("steering"))
            thr = _first_float(s.get("actual_throttle"), s.get("model_throttle"), info.get("throttle"))
            if steer is not None and thr is not None:
                model_pair = [float(steer), float(thr)]
        if model_pair is not None:
            actions.append(model_pair)

        pid_pair = _extract_pair_from_step(s, ["pid_action", "pid_actions", "expert_action", "expert_actions", "target_action", "target_actions"])
        if pid_pair is None:
            pid_steer = _first_float(
                s.get("pid_steer"), s.get("expert_steer"), s.get("target_steer"),
                info.get("pid_steer"), info.get("expert_steer"), info.get("target_steer"),
                pid_state.get("pid_steer"), pid_state.get("expert_steer"), pid_state.get("target_steer"),
                pid_state.get("steer"), pid_state.get("steering"),
            )
            pid_thr = _first_float(
                s.get("pid_throttle"), s.get("expert_throttle"), s.get("target_throttle"),
                info.get("pid_throttle"), info.get("expert_throttle"), info.get("target_throttle"),
                pid_state.get("pid_throttle"), pid_state.get("expert_throttle"), pid_state.get("target_throttle"),
                pid_state.get("throttle"),
            )
            if pid_steer is not None and pid_thr is not None:
                pid_pair = [float(pid_steer), float(pid_thr)]
        if pid_pair is not None:
            pid_actions.append(pid_pair)

        last_info = info

    is_success_val = last_info.get("is_success", last_info.get("isSuccess", 0))
    is_success = bool(int(is_success_val) == 1) if isinstance(is_success_val, (int, float)) else False
    timeout = bool(last_info.get("timeout", False))

    return {
        "xte": cte_seq,
        "angle_diff": ang_seq,
        "actions": actions,
        "pid_actions": pid_actions,
        "isSuccess": is_success,
        "timeout": timeout,
        "_pd_steps": steps,
    }


def _looks_like_step_dict(d: Dict[str, Any]) -> bool:
    if "info" in d or "pid_state" in d:
        return True
    if "step" in d:
        return True
    for k in (
        "pid_steer", "pid_throttle",
        "expert_steer", "expert_throttle",
        "target_steer", "target_throttle",
        "model_steer", "model_throttle",
        "actual_steer", "actual_throttle",
    ):
        if k in d:
            return True
    return False


def _looks_like_step_list(xs: Any) -> bool:
    if not isinstance(xs, list) or not xs:
        return False
    if not isinstance(xs[0], dict):
        return False
    return _looks_like_step_dict(xs[0])


def _coerce_log_entries(log_json: Any) -> List[Dict[str, Any]]:
    # Case A: list of per-step dicts -> treat as a PD-style step list
    if _looks_like_step_list(log_json):
        one = _parse_pd_log({"steps": log_json})
        return [one] if isinstance(one, dict) else []

    # Case B: list of entry dicts (Jungle segments, or multiple entries)
    if isinstance(log_json, list):
        return [x for x in log_json if isinstance(x, dict)]

    # Case C: dict wrapper around entries
    if isinstance(log_json, dict):
        for k in ("entries", "log_entries"):
            v = log_json.get(k)
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                return [x for x in v if isinstance(x, dict)]

        # Case D: dict with "steps"
        if "steps" in log_json:
            one = _parse_pd_log(log_json)
            return [one] if isinstance(one, dict) else []

        # Case E: single entry dict
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
    
    # Configuration-based log selection instead of path-based detection
    # The 'use_new_log' flag should be passed from the job configuration
    use_new_log = paired_severities  # Use paired_severities as proxy for now
    
    for ep in episodes:
        if not isinstance(ep, dict):
            continue

        ep_id = safe_str(ep.get("id"), default="unknown")
        ep_dir = run_dir / "episodes" / ep_id

        # Define all possible log paths
        log_rel = safe_str(ep.get("log"), default=f"episodes/{ep_id}/log.json")
        log_path = run_dir / log_rel
        pd_log_path = run_dir / "episodes" / ep_id / "pd_log.json"
        new_log_path = run_dir / "episodes" / ep_id / "new_log.json"

        log_json = None

        # Load log with explicit priority order
        if use_new_log:
            # For special cases (like GenRoads with paired severities): prefer new_log.json
            for path_to_try in [new_log_path, pd_log_path, log_path]:
                if path_to_try.exists():
                    log_json = try_read_json(path_to_try)
                    if log_json is not None:
                        break
        else:
            # For standard cases: use legacy priority (pd_log, then log)
            for path_to_try in [pd_log_path, log_path]:
                if path_to_try.exists():
                    log_json = try_read_json(path_to_try)
                    if log_json is not None:
                        break

        meta_path = run_dir / "episodes" / ep_id / "meta.json"
        events_path = run_dir / "episodes" / ep_id / "events.json"

        meta = try_read_json(meta_path)
        if not isinstance(meta, dict):
            meta = {}

        events = try_read_json(events_path)
        if not isinstance(events, list):
            events = []

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


def _coerce_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, str):
            x = x.strip()
        return int(x)
    except Exception:
        return default


def _severity_for_entry(sev_raw: Any, idx: int, n_entries: int, *, paired_severities: bool) -> int:
    if not paired_severities or not isinstance(sev_raw, list):
        return _coerce_int(sev_raw, default=0)

    sev_list = [_coerce_int(s, default=0) for s in sev_raw]
    if not sev_list:
        return 0

    if n_entries > 0 and len(sev_list) == n_entries and 0 <= idx < len(sev_list):
        return int(sev_list[idx])

    if len(sev_list) == 1:
        return int(sev_list[0])
    if 0 <= idx < len(sev_list):
        return int(sev_list[idx])

    return int(sev_list[0])


def normalize_udacity_entry(
    raw: Dict[str, Any],
    map_name: str,
    test_type: str,
    baseline_name: str = BASELINE_NAME_DEFAULT,
    paired_severities: bool = False,
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

    pert_raw = ep_m.get("perturbation")
    pert_s = "" if pert_raw is None else str(pert_raw).strip()
    is_baseline = (pert_raw is None) or (pert_s == "") or (pert_s.lower() == "none")
    if is_baseline:
        perturbation = baseline_name
        sev_raw = 0
    else:
        perturbation = safe_str(pert_raw, default=baseline_name)
        sev_raw = ep_m.get("severity", 0)

    road = safe_str(ep_m.get("road"), default=map_name)

    seg_ids_list = meta.get("segs") if isinstance(meta.get("segs"), list) else []
    num_segments = infer_num_segments_from_snapshot(config_snapshot)
    if num_segments <= 0 and map_name.lower() == "jungle":
        num_segments = len(seg_ids_list)

    out: List[Dict[str, Any]] = []
    n_entries = len(log_entries)

    for idx, entry in enumerate(log_entries):
        severity = 0 if is_baseline else _severity_for_entry(sev_raw, idx, n_entries, paired_severities=paired_severities)

        # Tracking series
        xte = entry.get("xte")
        if xte is None:
            xte = entry.get("cte")
        if xte is None:
            xte = entry.get("xte_seq")
        if xte is None:
            xte = entry.get("cte_seq")
        xte_list = xte if isinstance(xte, list) else _as_float_list(xte)

        angle_err = entry.get("angle_diff") if "angle_diff" in entry else None
        if angle_err is None:
            angle_err = entry.get("angle_errors")
        if angle_err is None:
            angle_err = entry.get("angle_err")
        angle_list = angle_err if isinstance(angle_err, list) else _as_float_list(angle_err)

        # Model actions (try several possible containers)
        actions = _normalize_action_list(
            entry.get("actions")
            or entry.get("model_actions")
            or entry.get("action")
            or {"steer": entry.get("steer") or entry.get("steering") or entry.get("model_steer") or entry.get("actual_steer"),
                "throttle": entry.get("throttle") or entry.get("thr") or entry.get("model_throttle") or entry.get("actual_throttle")}
        )

        # PID/expert actions (robust: pid_actions, pid_action, expert/target, or pid_steer+pid_throttle arrays)
        pid_actions = _normalize_action_list(
            entry.get("pid_actions")
            or entry.get("pid_action")
            or entry.get("expert_actions")
            or entry.get("expert_action")
            or entry.get("target_actions")
            or entry.get("target_action")
        )
        if not pid_actions:
            pid_actions = _normalize_action_list({
                "steer": entry.get("pid_steer") or entry.get("expert_steer") or entry.get("target_steer"),
                "throttle": entry.get("pid_throttle") or entry.get("expert_throttle") or entry.get("target_throttle"),
            })

        timeout = bool(entry.get("timeout", False))

        if map_name.lower() == "jungle":
            seg_id = safe_str(seg_ids_list[idx], default=f"segment_{idx:02d}") if idx < len(seg_ids_list) else f"segment_{idx:02d}"
            task_id = seg_id
        else:
            task_id = road

        if map_name.lower() == "jungle":
            seg_failed = has_failure_for_task(events=events, task_id=task_id)
            is_success = (not seg_failed) and (not timeout)
        else:
            is_success = bool(entry.get("isSuccess", entry.get("is_success", False)))

        series_len = int(min(len(xte_list), len(angle_list)) if (xte_list and angle_list) else (len(xte_list) or len(angle_list) or 0))

        if map_name.lower() == "jungle":
            ev_step = first_failure_step_for_task(events=events, task_id=task_id)
        else:
            steps = entry.get("_pd_steps") if isinstance(entry, dict) else None
            ev_step = first_failure_step_from_pd_steps(steps if isinstance(steps, list) else None)

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
            "severity": int(severity),
            "is_success": is_success,
            "timeout": timeout,
            "xte": xte_list,
            "angle_err": angle_list,
            "actions": actions,
            "pid_actions": pid_actions,
            "num_segments": num_segments,
            "fail_stop_idx": int(fail_stop_idx),
            "track_error_name": "cte" if map_name.lower() != "jungle" else "xte",
        })

    return out