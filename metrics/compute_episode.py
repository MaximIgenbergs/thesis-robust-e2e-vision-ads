from __future__ import annotations

from typing import Any, Dict

from metrics.metrics import tracking_metrics, action_deviation_to_pid


def compute_udacity_entry_metrics(ep: Dict[str, Any]) -> Dict[str, Any]:
    track = tracking_metrics(ep["xte"], ep["angle_err"])
    dev = action_deviation_to_pid(ep["actions"], ep["pid_actions"])

    return {
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
        # tracking (episode-entry level)
        "xte_abs_p95": track["xte_abs_p95"],
        "angle_abs_p95": track["angle_abs_p95"],
        # PID vs model
        "pid_mae_steer": dev["mae_steer"],
        "pid_mae_throttle": dev["mae_throttle"],
        "pid_dev_mean": dev["dev_mean"],
        "pid_dev_p95": dev["dev_p95"],
        "pid_n": dev["n"],
    }
