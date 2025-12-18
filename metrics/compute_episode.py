from __future__ import annotations

from typing import Any, Dict

from metrics.metrics import tracking_metrics, action_deviation_to_pid


def compute_udacity_entry_metrics(ep: Dict[str, Any]) -> Dict[str, Any]:
    stop_idx = ep.get("fail_stop_idx", None)

    track_full = tracking_metrics(ep["xte"], ep["angle_err"], stop_idx=None)
    track_pre = tracking_metrics(ep["xte"], ep["angle_err"], stop_idx=stop_idx)

    dev_full = action_deviation_to_pid(ep["actions"], ep["pid_actions"], stop_idx=None)
    dev_pre = action_deviation_to_pid(ep["actions"], ep["pid_actions"], stop_idx=stop_idx)

    row = {
        "sim": ep["sim"],
        "map": ep["map"],
        "test_type": ep["test_type"],
        "model": ep["model"],
        "run_id": ep["run_id"],
        "run_ts": ep["run_ts"],
        "entry_id": ep["entry_id"],
        "episode_folder": ep["episode_folder"],
        "entry_index": int(ep["entry_index"]),
        "task_id": ep["task_id"],
        "road": ep["road"],
        "perturbation": ep["perturbation"],
        "severity": int(ep["severity"]),
        "is_success": int(bool(ep["is_success"])),
        "timeout": int(bool(ep["timeout"])),
        "num_segments": int(ep.get("num_segments", 0)),
        "fail_stop_idx": int(ep.get("fail_stop_idx", 0)),

        # tracking (full)
        "xte_abs_p95": track_full["xte_abs_p95"],
        "angle_abs_p95": track_full["angle_abs_p95"],
        "xte_abs_mean": track_full["xte_abs_mean"],
        "angle_abs_mean": track_full["angle_abs_mean"],

        # tracking (pre-fail)
        "xte_abs_p95_pre": track_pre["xte_abs_p95"],
        "angle_abs_p95_pre": track_pre["angle_abs_p95"],
        "xte_abs_mean_pre": track_pre["xte_abs_mean"],
        "angle_abs_mean_pre": track_pre["angle_abs_mean"],

        # PID deviation (full)
        "pid_mae_steer": dev_full["mae_steer"],
        "pid_mae_throttle": dev_full["mae_throttle"],
        "pid_dev_mean": dev_full["dev_mean"],
        "pid_dev_p95": dev_full["dev_p95"],
        "pid_n": dev_full["n"],

        # PID deviation (pre-fail)
        "pid_mae_steer_pre": dev_pre["mae_steer"],
        "pid_mae_throttle_pre": dev_pre["mae_throttle"],
        "pid_dev_mean_pre": dev_pre["dev_mean"],
        "pid_dev_p95_pre": dev_pre["dev_p95"],
        "pid_n_pre": dev_pre["n"],

        "track_error_name": ep.get("track_error_name", "xte"),
    }

    # GenRoads uses CTE; expose explicit CTE fields as aliases
    if str(ep.get("track_error_name", "")).lower() == "cte":
        row.update({
            "cte_abs_p95": row["xte_abs_p95"],
            "cte_abs_mean": row["xte_abs_mean"],
            "cte_abs_p95_pre": row["xte_abs_p95_pre"],
            "cte_abs_mean_pre": row["xte_abs_mean_pre"],
        })

    return row
