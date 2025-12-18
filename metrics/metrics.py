from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Optional

import numpy as np


# ----------------------------
# Small numeric helpers
# ----------------------------

def seq_to_np(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _finite_mean(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_percentile(arr: np.ndarray, q: float) -> float:
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _clip_stop(stop_idx: Optional[int], n: int) -> int:
    if stop_idx is None:
        return n
    try:
        s = int(stop_idx)
    except Exception:
        return n
    return max(0, min(n, s))


def _slice(xs: Sequence[float], stop_idx: Optional[int]) -> Sequence[float]:
    n = len(xs) if xs is not None else 0
    s = _clip_stop(stop_idx, n)
    return xs[:s]


def summarize_abs(x: Sequence[float], qs: Sequence[float] = (50.0, 95.0)) -> Dict[str, float]:
    if not x:
        out = {f"abs_p{int(q)}": float("nan") for q in qs}
        out.update({"abs_mean": float("nan"), "abs_max": float("nan")})
        return out
    arr = np.abs(seq_to_np(x))
    out: Dict[str, float] = {
        "abs_mean": float(np.mean(arr)),
        "abs_max": float(np.max(arr)),
    }
    for q in qs:
        out[f"abs_p{int(q)}"] = float(np.percentile(arr, q))
    return out


# ----------------------------
# CARLA summaries (route-level)
# ----------------------------

@dataclass(frozen=True)
class CarlaEpisodeSummary:
    driving_score: float
    blocked: bool
    time_to_block_s: Optional[float] = None
    dist_to_block_m: Optional[float] = None


def carla_blocked_rate(episodes: Sequence[CarlaEpisodeSummary]) -> float:
    if not episodes:
        return float("nan")
    blocked = sum(1 for e in episodes if e.blocked)
    return blocked / float(len(episodes))


def carla_ds_active(episodes: Sequence[CarlaEpisodeSummary]) -> Dict[str, float]:
    n_total = len(episodes)
    active = [e.driving_score for e in episodes if not e.blocked]
    arr = seq_to_np(active) if active else np.asarray([], dtype=np.float64)
    return {
        "n_total": float(n_total),
        "n_active": float(len(active)),
        "blocked_rate": carla_blocked_rate(episodes),
        "ds_active_mean": _finite_mean(arr),
        "ds_active_p50": _finite_percentile(arr, 50.0),
        "ds_active_p95": _finite_percentile(arr, 95.0),
    }


def carla_ds_all(episodes: Sequence[CarlaEpisodeSummary]) -> Dict[str, float]:
    ds = [e.driving_score for e in episodes]
    arr = seq_to_np(ds) if ds else np.asarray([], dtype=np.float64)
    return {
        "n_total": float(len(ds)),
        "ds_all_mean": _finite_mean(arr),
        "ds_all_p50": _finite_percentile(arr, 50.0),
        "ds_all_p95": _finite_percentile(arr, 95.0),
    }


def carla_time_to_block(episodes: Sequence[CarlaEpisodeSummary]) -> Dict[str, float]:
    ttb = [e.time_to_block_s for e in episodes if e.blocked and e.time_to_block_s is not None]
    dtb = [e.dist_to_block_m for e in episodes if e.blocked and e.dist_to_block_m is not None]
    t = seq_to_np(ttb) if ttb else np.asarray([], dtype=np.float64)
    d = seq_to_np(dtb) if dtb else np.asarray([], dtype=np.float64)
    return {
        "ttb_mean_s": _finite_mean(t),
        "ttb_p50_s": _finite_percentile(t, 50.0),
        "ttb_p95_s": _finite_percentile(t, 95.0),
        "dtb_mean_m": _finite_mean(d),
        "dtb_p50_m": _finite_percentile(d, 50.0),
        "dtb_p95_m": _finite_percentile(d, 95.0),
    }


# ----------------------------
# Udacity: pass metrics
# ----------------------------

def pass_rate(is_success_flags: Sequence[bool]) -> float:
    if not is_success_flags:
        return float("nan")
    return float(np.mean(np.asarray(is_success_flags, dtype=np.float64)))


def segments_passed(segment_success_flags: Sequence[bool]) -> Tuple[int, float]:
    n = len(segment_success_flags)
    if n == 0:
        return 0, float("nan")
    k = sum(1 for s in segment_success_flags if bool(s))
    return k, k / float(n)


# ----------------------------
# Tracking metrics (full + pre-fail)
# ----------------------------

def tracking_metrics(
    xte: Sequence[float],
    angle_err: Sequence[float],
    stop_idx: Optional[int] = None,
) -> Dict[str, float]:
    xte_s = _slice(xte, stop_idx)
    ang_s = _slice(angle_err, stop_idx)

    x = summarize_abs(xte_s)
    a = summarize_abs(ang_s)

    return {
        "xte_abs_p95": x["abs_p95"],
        "angle_abs_p95": a["abs_p95"],
        # optional, useful for analysis:
        "xte_abs_mean": x["abs_mean"],
        "angle_abs_mean": a["abs_mean"],
        "xte_abs_max": x["abs_max"],
        "angle_abs_max": a["abs_max"],
        "n": float(min(len(xte_s), len(ang_s))) if (xte_s and ang_s) else float(len(xte_s) or len(ang_s) or 0),
    }


# ----------------------------
# PID vs model actions (full + pre-fail)
# ----------------------------

def _split_actions(actions: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not actions:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"actions must be Nx2+, got shape {arr.shape}")
    return arr[:, 0], arr[:, 1]


def action_deviation_to_pid(
    actions: Sequence[Sequence[float]],
    pid_actions: Sequence[Sequence[float]],
    stop_idx: Optional[int] = None,
    w_steer: float = 1.0,
    w_throttle: float = 1.0,
) -> Dict[str, float]:
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

    n = int(min(len(s), len(ps), len(t), len(pt)))
    n = _clip_stop(stop_idx, n)
    if n <= 0:
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
    if not np.isfinite(clean) or clean <= 0.0:
        return float("nan")
    return float(1.0 - (corrupted / clean))


def auc_over_severity(values: Sequence[float], severities: Sequence[float]) -> float:
    n = min(len(values), len(severities))
    if n < 2:
        return float("nan")
    xs = np.asarray(severities[:n], dtype=np.float64)
    ys = np.asarray(values[:n], dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return float("nan")
    return float(np.trapz(ys, xs))


def corruption_error(rel_drops_over_severity: Sequence[float]) -> float:
    if not rel_drops_over_severity:
        return float("nan")
    arr = np.asarray(rel_drops_over_severity, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def mean_corruption_error(per_corruption_errors: Sequence[float]) -> float:
    if not per_corruption_errors:
        return float("nan")
    arr = np.asarray(per_corruption_errors, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


# ----------------------------
# Generalization helper
# ----------------------------

def seen_unseen_gap(seen: float, unseen: float) -> float:
    if not (np.isfinite(seen) and np.isfinite(unseen)):
        return float("nan")
    return float(unseen - seen)
