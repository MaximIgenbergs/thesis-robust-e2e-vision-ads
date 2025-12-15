from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from metrics.metrics import (
    auc_over_severity,
    relative_drop,
    corruption_error,
    mean_corruption_error,
    CarlaEpisodeSummary,
    carla_ds_active,
)


def _mean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def aggregate_udacity_summary(entry_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Udacity:
      - Jungle primary: mean_segments_passed_rate_per_run (run-level)
      - GenRoads primary: pass_rate
    Always also reports: xte_abs_p95_mean, angle_abs_p95_mean, pid_dev/pid_mae means.
    """
    groups: Dict[Tuple[str, str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in entry_rows:
        key = (r["map"], r["test_type"], r["model"], r["perturbation"], int(r["severity"]))
        groups[key].append(r)

    out: List[Dict[str, Any]] = []

    for (map_name, test_type, model, perturbation, severity), rows in sorted(groups.items()):
        is_jungle = str(map_name).lower() == "jungle"

        success = [float(r["is_success"]) for r in rows]
        xte_p95 = [float(r["xte_abs_p95"]) for r in rows]
        ang_p95 = [float(r["angle_abs_p95"]) for r in rows]
        pid_dev_mean = [float(r["pid_dev_mean"]) for r in rows]
        pid_dev_p95 = [float(r["pid_dev_p95"]) for r in rows]
        pid_mae_s = [float(r["pid_mae_steer"]) for r in rows]
        pid_mae_t = [float(r["pid_mae_throttle"]) for r in rows]

        base = {
            "sim": "udacity",
            "map": map_name,
            "test_type": test_type,
            "model": model,
            "perturbation": perturbation,
            "severity": int(severity),
            "n_entries": len(rows),
            "xte_abs_p95_mean": _mean(xte_p95),
            "angle_abs_p95_mean": _mean(ang_p95),
            "pid_dev_mean_mean": _mean(pid_dev_mean),
            "pid_dev_p95_mean": _mean(pid_dev_p95),
            "pid_mae_steer_mean": _mean(pid_mae_s),
            "pid_mae_throttle_mean": _mean(pid_mae_t),
        }

        if not is_jungle:
            base["primary_metric"] = "pass_rate"
            base["pass_rate"] = _mean(success)
            out.append(base)
            continue

        # Jungle: each row is a segment-entry attempt. We need run-level segments passed.
        # Here you are computing for ONE run dir, but keep it general.
        segment_pass_rate_micro = _mean(success)

        per_run_seen: Dict[str, set] = defaultdict(set)
        per_run_passed: Dict[str, set] = defaultdict(set)
        per_run_num_segments: Dict[str, int] = {}

        for r in rows:
            run_id = str(r["run_id"])
            seg_id = str(r["task_id"])
            per_run_seen[run_id].add(seg_id)
            if int(r["is_success"]) == 1:
                per_run_passed[run_id].add(seg_id)

            ns = int(r.get("num_segments", 0))
            if ns > 0:
                per_run_num_segments[run_id] = ns

        seg_pass_counts: List[float] = []
        seg_pass_rates: List[float] = []

        for run_id in per_run_seen.keys():
            k = len(per_run_passed.get(run_id, set()))
            denom = per_run_num_segments.get(run_id, 0)
            if denom <= 0:
                denom = len(per_run_seen[run_id])
            seg_pass_counts.append(float(k))
            seg_pass_rates.append(float(k) / float(denom) if denom > 0 else float("nan"))

        base["primary_metric"] = "mean_segments_passed_rate_per_run"
        base["segment_pass_rate_micro"] = segment_pass_rate_micro
        base["mean_segments_passed_per_run"] = _mean(seg_pass_counts)
        base["mean_segments_passed_rate_per_run"] = _mean(seg_pass_rates)
        base["n_runs"] = len(per_run_seen)

        out.append(base)

    return out


def aggregate_carla_summary(route_rows: list[dict]) -> list[dict]:
    """
    Aggregates per (model, test_type, condition, perturbation, severity).
    Produces both DS including blocked routes and DS excluding blocked routes.
    """
    from collections import defaultdict
    import math

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in route_rows:
        key = (
            r["model"],
            r["test_type"],
            r["condition"],
            r["perturbation"],
            int(r["severity"]),
        )
        groups[key].append(r)

    out: list[dict] = []
    for (model, test_type, condition, perturbation, severity), rows in sorted(groups.items()):
        ds_all = [float(r["driving_score"]) for r in rows if r.get("driving_score") is not None]
        ds_active = [float(r["driving_score"]) for r in rows if r.get("driving_score") is not None and int(r.get("blocked", 0)) == 0]

        blocked_flags = [int(r.get("blocked", 0)) for r in rows]
        blocked_rate = (sum(blocked_flags) / len(blocked_flags)) if blocked_flags else 0.0

        # time-to-block (TTB): duration_game_s for blocked routes only (optional)
        ttb = [
            float(r["duration_game_s"])
            for r in rows
            if int(r.get("blocked", 0)) == 1 and r.get("duration_game_s") is not None
        ]

        def mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        out.append({
            "sim": "carla",
            "map": "multi-town",
            "test_type": test_type,
            "model": model,
            "condition": condition,
            "perturbation": perturbation,
            "severity": int(severity),

            # advisor-required:
            "ds_all_mean": mean(ds_all),

            # your diagnostic:
            "ds_active_mean": mean(ds_active),

            "blocked_rate": blocked_rate,

            # optional diagnostics (keep or drop later):
            "n_routes": len(rows),
            "n_blocked": sum(blocked_flags),
            "ttb_mean_s": mean(ttb),
        })

    return out



def robustness_summaries(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Creates per-(sim,map,test_type,model,perturbation) robustness aggregates over severity:
      - AUC(primary vs severity)
      - CE (relative-drop vs baseline) if baseline row exists
    """
    def primary_value(row: Dict[str, Any]) -> float:
        if row["sim"] == "udacity":
            if str(row.get("map", "")).lower() == "jungle":
                return float(row.get("mean_segments_passed_rate_per_run", np.nan))
            return float(row.get("pass_rate", np.nan))
        # carla
        return float(row.get("ds_active_mean", np.nan))

    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        groups[(r["sim"], r.get("map", ""), r["test_type"], r["model"], r["perturbation"])].append(r)

    # baseline lookup per (sim,map,test_type,model)
    baseline_by_setting: Dict[Tuple[str, str, str, str], float] = {}
    for r in summary_rows:
        pert = str(r.get("perturbation", "")).lower()
        if pert == "baseline":
            baseline_by_setting[(r["sim"], r.get("map", ""), r["test_type"], r["model"])] = primary_value(r)

    out: List[Dict[str, Any]] = []
    for (sim, map_name, test_type, model, perturbation), rows in sorted(groups.items()):
        rows_sorted = sorted(rows, key=lambda x: int(x.get("severity", 0)))
        severities = [float(r.get("severity", 0)) for r in rows_sorted]
        values = [primary_value(r) for r in rows_sorted]

        auc = auc_over_severity(values, severities)

        clean_val = baseline_by_setting.get((sim, map_name, test_type, model), float("nan"))
        rel_drops = [relative_drop(clean_val, v) for v in values] if np.isfinite(clean_val) else []
        ce = corruption_error(rel_drops) if rel_drops else float("nan")

        out.append({
            "sim": sim,
            "map": map_name,
            "test_type": test_type,
            "model": model,
            "perturbation": perturbation,
            "robust_auc": auc,
            "robust_ce": ce,
            "baseline_primary": clean_val,
            "n_points": len(values),
        })

    return out


def mce_over_all_corruptions(robustness_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for r in robustness_rows:
        pert = str(r.get("perturbation", "")).lower()
        if pert == "baseline":
            continue
        groups[(r["sim"], r.get("map", ""), r["test_type"], r["model"])].append(float(r.get("robust_ce", np.nan)))

    out: List[Dict[str, Any]] = []
    for (sim, map_name, test_type, model), ces in sorted(groups.items()):
        out.append({
            "sim": sim,
            "map": map_name,
            "test_type": test_type,
            "model": model,
            "mce_over_corruptions": mean_corruption_error(ces),
            "n_corruptions": len(ces),
        })
    return out
