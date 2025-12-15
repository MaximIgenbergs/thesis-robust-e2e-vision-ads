from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def _mean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def aggregate_udacity_table(episode_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Per-(map,test_type,model,perturbation,severity) rows.

    GenRoads primary: pass_rate.
    Jungle primary: mean_segments_passed_per_run and mean_segments_passed_rate_per_run,
                    using per-run num_segments inferred from config_snapshot.
    """
    groups: Dict[Tuple[str, str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in episode_rows:
        key = (r["map"], r["test_type"], r["model"], r["perturbation"], int(r["severity"]))
        groups[key].append(r)

    out: List[Dict[str, Any]] = []

    for (map_name, test_type, model, perturbation, severity), rows in sorted(groups.items()):
        is_jungle = str(map_name).lower() == "jungle"

        success_flags = [float(r["is_success"]) for r in rows]
        xte_p95s = [float(r["xte_abs_p95"]) for r in rows]
        ang_p95s = [float(r["angle_abs_p95"]) for r in rows]

        pid_dev_means = [float(r["pid_dev_mean"]) for r in rows]
        pid_dev_p95s = [float(r["pid_dev_p95"]) for r in rows]
        pid_mae_steer = [float(r["pid_mae_steer"]) for r in rows]
        pid_mae_thr = [float(r["pid_mae_throttle"]) for r in rows]

        base = {
            "sim": "udacity",
            "map": map_name,
            "test_type": test_type,
            "model": model,
            "perturbation": perturbation,
            "severity": int(severity),
            "n_episodes": len(rows),
            "xte_abs_p95_mean": _mean(xte_p95s),
            "angle_abs_p95_mean": _mean(ang_p95s),
            "pid_dev_mean_mean": _mean(pid_dev_means),
            "pid_dev_p95_mean": _mean(pid_dev_p95s),
            "pid_mae_steer_mean": _mean(pid_mae_steer),
            "pid_mae_throttle_mean": _mean(pid_mae_thr),
        }

        if not is_jungle:
            base["primary_metric"] = "pass_rate"
            base["pass_rate"] = _mean(success_flags)
            out.append(base)
            continue

        # Jungle: each row is a segment attempt; also compute run-level segments passed.
        segment_pass_rate_micro = _mean(success_flags)

        per_run_seen: Dict[str, set] = defaultdict(set)
        per_run_passed: Dict[str, set] = defaultdict(set)
        per_run_num_segments: Dict[str, int] = {}

        for r in rows:
            run_ts = str(r["run_ts"])
            seg_id = str(r["task_id"])

            per_run_seen[run_ts].add(seg_id)
            if int(r["is_success"]) == 1:
                per_run_passed[run_ts].add(seg_id)

            ns = int(r.get("num_segments", 0))
            if ns > 0:
                per_run_num_segments[run_ts] = ns

        segments_passed_counts: List[float] = []
        segments_passed_rates: List[float] = []

        for run_ts in per_run_seen.keys():
            k = len(per_run_passed.get(run_ts, set()))
            segments_passed_counts.append(float(k))

            denom = per_run_num_segments.get(run_ts, 0)
            if denom <= 0:
                # fallback: use "seen unique segments in this run"
                denom = len(per_run_seen[run_ts])
            segments_passed_rates.append(float(k) / float(denom) if denom > 0 else float("nan"))

        base["primary_metric"] = "mean_segments_passed_per_run"
        base["segment_pass_rate_micro"] = segment_pass_rate_micro
        base["mean_segments_passed_per_run"] = _mean(segments_passed_counts)
        base["mean_segments_passed_rate_per_run"] = _mean(segments_passed_rates)
        base["n_runs"] = len(per_run_seen)

        out.append(base)

    return out
