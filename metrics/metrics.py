from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Optional

import numpy as np


def seq_to_np(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def mean(x: Sequence[float]) -> float:
    if not x:
        return float("nan")
    return float(np.mean(seq_to_np(x)))


def percentile(x: Sequence[float], q: float) -> float:
    if not x:
        return float("nan")
    return float(np.percentile(seq_to_np(x), q))


def abs_percentile(x: Sequence[float], q: float) -> float:
    if not x:
        return float("nan")
    arr = np.abs(seq_to_np(x))
    return float(np.percentile(arr, q))


def summarize_abs(x: Sequence[float], qs: Sequence[float] = (50.0, 95.0)) -> Dict[str, float]:
    """
    Summary stats for absolute values of a signal.
    """
    if not x:
        return {f"abs_p{int(q)}": float("nan") for q in qs} | {"abs_mean": float("nan"), "abs_max": float("nan")}
    arr = np.abs(seq_to_np(x))
    out: Dict[str, float] = {"abs_mean": float(np.mean(arr)), "abs_max": float(np.max(arr))}
    for q in qs:
        out[f"abs_p{int(q)}"] = float(np.percentile(arr, q))
    return out


# ----------------------------
# CARLA (Leaderboard) summaries
# ----------------------------

@dataclass(frozen=True)
class CarlaEpisodeSummary:
    driving_score: float
    blocked: bool
    # Optional diagnostics (only used if passed)
    time_to_block_s: Optional[float] = None
    dist_to_block_m: Optional[float] = None


def carla_blocked_rate(episodes: Sequence[CarlaEpisodeSummary]) -> float:
    if not episodes:
        return float("nan")
    blocked = sum(1 for e in episodes if e.blocked)
    return blocked / float(len(episodes))


def carla_ds_active(episodes: Sequence[CarlaEpisodeSummary]) -> Dict[str, float]:
    """
    Main CARLA metric per your decision:
      - DS_active is computed on NON-blocked episodes only.
      - Always report BR and N_active so the filtering is transparent.
    """
    n_total = len(episodes)
    active = [e.driving_score for e in episodes if not e.blocked]

    out: Dict[str, float] = {
        "n_total": float(n_total),
        "n_active": float(len(active)),
        "blocked_rate": carla_blocked_rate(episodes),
        "ds_active_mean": float(np.mean(active)) if active else float("nan"),
        "ds_active_p50": float(np.percentile(seq_to_np(active), 50.0)) if active else float("nan"),
        "ds_active_p95": float(np.percentile(seq_to_np(active), 95.0)) if active else float("nan"),
    }
    return out


def carla_time_to_block(episodes: Sequence[CarlaEpisodeSummary]) -> Dict[str, float]:
    """
    Optional diagnostics. Only meaningful if time_to_block_s/dist_to_block_m are provided.
    """
    ttb = [e.time_to_block_s for e in episodes if e.blocked and e.time_to_block_s is not None]
    dtb = [e.dist_to_block_m for e in episodes if e.blocked and e.dist_to_block_m is not None]

    return {
        "ttb_mean_s": mean(ttb) if ttb else float("nan"),
        "ttb_p50_s": percentile(ttb, 50.0) if ttb else float("nan"),
        "ttb_p95_s": percentile(ttb, 95.0) if ttb else float("nan"),
        "dtb_mean_m": mean(dtb) if dtb else float("nan"),
        "dtb_p50_m": percentile(dtb, 50.0) if dtb else float("nan"),
        "dtb_p95_m": percentile(dtb, 95.0) if dtb else float("nan"),
    }


# ----------------------------
# Udacity: pass metrics
# ----------------------------

def pass_rate(is_success_flags: Sequence[bool]) -> float:
    """
    Mean success over episodes/roads.
    """
    if not is_success_flags:
        return float("nan")
    return float(np.mean(np.asarray(is_success_flags, dtype=np.float64)))


def segments_passed(segment_success_flags: Sequence[bool]) -> Tuple[int, float]:
    """
    For Jungle runs:
      - per run: segments_passed in [0..num_segments]
      - you can aggregate later via mean over runs or micro-average over all segments.
    """
    n = len(segment_success_flags)
    if n == 0:
        return 0, float("nan")
    k = sum(1 for s in segment_success_flags if bool(s))
    return k, k / float(n)


# ----------------------------
# Udacity: tracking summaries
# ----------------------------

def tracking_metrics(xte: Sequence[float], angle_err: Sequence[float]) -> Dict[str, float]:
    """
    What you said you want to report:
      - p95 |xte|
      - p95 |angle error|
    """
    return {
        "xte_abs_p95": abs_percentile(xte, 95.0),
        "angle_abs_p95": abs_percentile(angle_err, 95.0),
        # optional but often handy in analysis/debug:
        "xte_abs_mean": summarize_abs(xte)["abs_mean"],
        "angle_abs_mean": summarize_abs(angle_err)["abs_mean"],
    }


# ----------------------------
# Udacity: PID vs model actions
# ----------------------------

def _split_actions(actions: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expects actions like [[steer, throttle], ...].
    Returns (steer, throttle) as float64 arrays.
    """
    if not actions:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"actions must be Nx2+, got shape {arr.shape}")
    return arr[:, 0], arr[:, 1]


def action_deviation_to_pid(
    actions: Sequence[Sequence[float]],
    pid_actions: Sequence[Sequence[float]],
    w_steer: float = 1.0,
    w_throttle: float = 1.0,
) -> Dict[str, float]:
    """
    Compare model controls to PID controls at matching timesteps.
    This is the metric your instructor suggested.

    Returns both:
      - interpretable per-channel MAE (steer/throttle)
      - combined weighted L2 deviation d_t and summaries (mean/p95)

    If pid_actions is empty, returns NaNs (so you notice missing expert logs).
    """
    if not pid_actions:
        return {
            "mae_steer": float("nan"),
            "mae_throttle": float("nan"),
            "dev_mean": float("nan"),
            "dev_p95": float("nan"),
            "n": 0.0,
        }

    s, t = _split_actions(actions)
    ps, pt = _split_actions(pid_actions)

    n = int(min(len(s), len(ps)))
    if n == 0:
        return {
            "mae_steer": float("nan"),
            "mae_throttle": float("nan"),
            "dev_mean": float("nan"),
            "dev_p95": float("nan"),
            "n": 0.0,
        }

    ds = s[:n] - ps[:n]
    dt = t[:n] - pt[:n]

    mae_s = float(np.mean(np.abs(ds)))
    mae_t = float(np.mean(np.abs(dt)))

    # combined per-frame deviation
    d = np.sqrt(w_steer * (ds ** 2) + w_throttle * (dt ** 2))

    return {
        "mae_steer": mae_s,
        "mae_throttle": mae_t,
        "dev_mean": float(np.mean(d)),
        "dev_p95": float(np.percentile(d, 95.0)),
        "n": float(n),
    }


# ----------------------------
# Robustness aggregation helpers
# ----------------------------

def relative_drop(clean: float, corrupted: float) -> float:
    """
    For higher-is-better metrics. Returns NaN if clean is invalid.
    """
    if not np.isfinite(clean) or clean <= 0.0:
        return float("nan")
    return float(1.0 - (corrupted / clean))


def auc_over_severity(values: Sequence[float], severities: Sequence[float]) -> float:
    """
    Trapezoidal AUC. Assumes severities are sorted ascending.
    Returns NaN if not enough points.
    """
    n = min(len(values), len(severities))
    if n < 2:
        return float("nan")

    xs = np.asarray(severities[:n], dtype=np.float64)
    ys = np.asarray(values[:n], dtype=np.float64)

    # If there are NaNs, drop those points
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return float("nan")

    return float(np.trapz(ys, xs))


def corruption_error(rel_drops_over_severity: Sequence[float]) -> float:
    """
    Mean relative drop for one corruption across severities.
    """
    if not rel_drops_over_severity:
        return float("nan")
    arr = np.asarray(rel_drops_over_severity, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def mean_corruption_error(per_corruption_errors: Sequence[float]) -> float:
    """
    Mean across corruptions.
    """
    if not per_corruption_errors:
        return float("nan")
    arr = np.asarray(per_corruption_errors, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


# ----------------------------
# Generalization helper
# ----------------------------

def seen_unseen_gap(seen: float, unseen: float) -> float:
    """
    unseen - seen (positive means better on unseen, usually negative).
    """
    if not (np.isfinite(seen) and np.isfinite(unseen)):
        return float("nan")
    return float(unseen - seen)
