from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from metrics.metrics import auc_over_severity, relative_drop, corruption_error, mean_corruption_error

_BASELINE_NAMES = {"baseline", "clean"}


def _mean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _is_baseline(name: Any) -> bool:
    if name is None:
        return True
    s = str(name).strip().lower()
    return (s == "") or (s in _BASELINE_NAMES) or (s == "none")


def aggregate_udacity_summary(entry_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in entry_rows:
        key = (r["map"], r["test_type"], r["model"], r["perturbation"], int(r["severity"]))
        groups[key].append(r)

    out: List[Dict[str, Any]] = []
    
    # Correct key has 5 elements: (map, test_type, model, perturbation, severity)
    for (map_name, test_type, model, perturbation, severity), rows in sorted(groups.items()):
        is_jungle = str(map_name).lower() == "jungle"

        success = [float(r["is_success"]) for r in rows]

        xte_p95 = [float(r["xte_abs_p95"]) for r in rows]
        xte_mean = [float(r["xte_abs_mean"]) for r in rows]
        ang_p95 = [float(r["angle_abs_p95"]) for r in rows]
        ang_mean = [float(r["angle_abs_mean"]) for r in rows]

        pid_dev_mean = [float(r["pid_dev_mean"]) for r in rows]
        pid_dev_p95 = [float(r["pid_dev_p95"]) for r in rows]
        pid_mae_s = [float(r["pid_mae_steer"]) for r in rows]
        pid_mae_t = [float(r["pid_mae_throttle"]) for r in rows]

        xte_p95_pre = [float(r["xte_abs_p95_pre"]) for r in rows]
        xte_mean_pre = [float(r["xte_abs_mean_pre"]) for r in rows]
        ang_p95_pre = [float(r["angle_abs_p95_pre"]) for r in rows]
        ang_mean_pre = [float(r["angle_abs_mean_pre"]) for r in rows]

        pid_dev_mean_pre = [float(r["pid_dev_mean_pre"]) for r in rows]
        pid_dev_p95_pre = [float(r["pid_dev_p95_pre"]) for r in rows]
        pid_mae_s_pre = [float(r["pid_mae_steer_pre"]) for r in rows]
        pid_mae_t_pre = [float(r["pid_mae_throttle_pre"]) for r in rows]

        base: Dict[str, Any] = {
            "sim": "udacity",
            "map": map_name,
            "test_type": test_type,
            "model": model,
            "perturbation": perturbation,
            "severity": int(severity),
            "n_entries": len(rows),
            "xte_abs_p95_mean": _mean(xte_p95),
            "xte_abs_mean_mean": _mean(xte_mean),
            "angle_abs_p95_mean": _mean(ang_p95),
            "angle_abs_mean_mean": _mean(ang_mean),
            "pid_dev_mean_mean": _mean(pid_dev_mean),
            "pid_dev_p95_mean": _mean(pid_dev_p95),
            "pid_mae_steer_mean": _mean(pid_mae_s),
            "pid_mae_throttle_mean": _mean(pid_mae_t),
            "xte_abs_p95_pre_mean": _mean(xte_p95_pre),
            "xte_abs_mean_pre_mean": _mean(xte_mean_pre),
            "angle_abs_p95_pre_mean": _mean(ang_p95_pre),
            "angle_abs_mean_pre_mean": _mean(ang_mean_pre),
            "pid_dev_mean_pre_mean": _mean(pid_dev_mean_pre),
            "pid_dev_p95_pre_mean": _mean(pid_dev_p95_pre),
            "pid_mae_steer_pre_mean": _mean(pid_mae_s_pre),
            "pid_mae_throttle_pre_mean": _mean(pid_mae_t_pre),
        }

        if not is_jungle:
            base["primary_metric"] = "pass_rate"
            base["pass_rate"] = _mean(success)
            base["cte_abs_p95_pre_mean"] = base["xte_abs_p95_pre_mean"]
            base["cte_abs_mean_pre_mean"] = base["xte_abs_mean_pre_mean"]
            base["cte_abs_p95_mean"] = base["xte_abs_p95_mean"]
            base["cte_abs_mean_mean"] = base["xte_abs_mean_mean"]
            out.append(base)
            continue

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
            denom = per_run_num_segments.get(run_id, 0) or len(per_run_seen[run_id])
            seg_pass_counts.append(float(k))
            seg_pass_rates.append(float(k) / float(denom) if denom > 0 else float("nan"))

        base["primary_metric"] = "mean_segments_passed_rate_per_run"
        base["segment_pass_rate_micro"] = segment_pass_rate_micro
        base["mean_segments_passed_per_run"] = _mean(seg_pass_counts)
        base["mean_segments_passed_rate_per_run"] = _mean(seg_pass_rates)
        base["n_runs"] = len(per_run_seen)

        out.append(base)

    return out


def aggregate_carla_summary(route_rows: list[dict], *, group_generalization_overall: bool = True) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in route_rows:
        test_type = str(r.get("test_type", ""))
        condition = str(r.get("condition", "unknown_condition"))
        if group_generalization_overall and test_type == "generalization":
            condition = "ALL"
        key = (r["model"], test_type, condition, r["perturbation"], int(r["severity"]))
        groups[key].append(r)

    out: list[dict] = []
    for (model, test_type, condition, perturbation, severity), rows in sorted(groups.items()):
        ds_all = [float(r["driving_score"]) for r in rows if r.get("driving_score") is not None]
        ds_nb = [float(r["driving_score"]) for r in rows if r.get("driving_score") is not None and int(r.get("blocked", 0)) == 0]

        blocked_flags = [int(r.get("blocked", 0)) for r in rows]
        blocked_rate = (sum(blocked_flags) / len(blocked_flags)) if blocked_flags else float("nan")

        ttb = [float(r["duration_game_s"]) for r in rows if int(r.get("blocked", 0)) == 1 and r.get("duration_game_s") is not None]

        def mean(xs: list[float]) -> float:
            arr = np.asarray(xs, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if arr.size else float("nan")

        n_routes = len(rows)
        n_blocked = int(sum(blocked_flags)) if blocked_flags else 0
        n_nb = int(len(ds_nb))

        out.append({
            "sim": "carla",
            "map": "multi-town",
            "test_type": test_type,
            "model": model,
            "condition": condition,
            "perturbation": perturbation,
            "severity": int(severity),
            "ds_all_mean": mean(ds_all),
            "blocked_rate": blocked_rate,
            "n_routes": n_routes,
            "n_blocked": n_blocked,
            "n_nb": n_nb,
            "ds_nb_mean": mean(ds_nb),
            "ttb_mean_s": mean(ttb),
        })

    return out


def robustness_summaries(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def primary_value(row: Dict[str, Any]) -> float:
        if row["sim"] == "udacity":
            if str(row.get("map", "")).lower() == "jungle":
                return float(row.get("mean_segments_passed_rate_per_run", np.nan))
            return float(row.get("pass_rate", np.nan))
        return float(row.get("ds_all_mean", np.nan))

    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        groups[(r["sim"], r.get("map", ""), r["test_type"], r["model"], r["perturbation"])].append(r)

    baseline_by_setting: Dict[Tuple[str, str, str, str], float] = {}
    for r in summary_rows:
        if _is_baseline(r.get("perturbation", "")) and int(r.get("severity", 0)) == 0:
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
        if _is_baseline(r.get("perturbation", "")):
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


def robustness_by_severity(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        groups[(r["sim"], r.get("map", ""), r["test_type"], r["model"], int(r.get("severity", 0)))].append(r)

    out: List[Dict[str, Any]] = []
    for (sim, map_name, test_type, model, severity), rows in sorted(groups.items()):
        rows_use: List[Dict[str, Any]]
        if severity == 0:
            rows_use = [r for r in rows if _is_baseline(r.get("perturbation", ""))] or rows
        else:
            rows_use = [r for r in rows if not _is_baseline(r.get("perturbation", ""))] or rows

        def mean_field(field: str) -> float:
            xs = []
            for r in rows_use:
                v = r.get(field, None)
                if v is None:
                    continue
                try:
                    xs.append(float(v))
                except Exception:
                    continue
            return _mean(xs)

        row: Dict[str, Any] = {
            "sim": sim,
            "map": map_name,
            "test_type": test_type,
            "model": model,
            "severity": int(severity),
            "n_conditions": len(rows_use),
        }

        if sim == "udacity":
            is_jungle = str(map_name).lower() == "jungle"
            if is_jungle:
                row["pr"] = mean_field("mean_segments_passed_rate_per_run")
            else:
                row["pr"] = mean_field("pass_rate")
            row["dt_mean"] = mean_field("pid_dev_mean_pre_mean")
            row["dt_p95"] = mean_field("pid_dev_p95_pre_mean")
            row["e_mean"] = mean_field("xte_abs_mean_pre_mean")
            row["e_p95"] = mean_field("xte_abs_p95_pre_mean")
        else:
            row["ds"] = mean_field("ds_all_mean")
            row["br"] = mean_field("blocked_rate")
            row["n_nb"] = mean_field("n_nb")
            row["ds_nb"] = mean_field("ds_nb_mean")

        out.append(row)

    return out


def rq1_robustness_wide(severity_rows: List[Dict[str, Any]], severities: Tuple[int, ...] = (0, 2, 4)) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in severity_rows:
        groups[(r["sim"], r.get("map", ""), r["model"])].append(r)

    out: List[Dict[str, Any]] = []
    for (sim, map_name, model), rows in sorted(groups.items()):
        by_sev = {int(r.get("severity", -1)): r for r in rows}
        base: Dict[str, Any] = {"sim": sim, "map": map_name, "model": model}

        for s in severities:
            r = by_sev.get(int(s), {})
            if sim == "udacity":
                base[f"sev{s}_PR"] = r.get("pr", float("nan"))
                base[f"sev{s}_E_dt"] = r.get("dt_mean", float("nan"))
                base[f"sev{s}_p95_dt"] = r.get("dt_p95", float("nan"))
                base[f"sev{s}_E_e"] = r.get("e_mean", float("nan"))
                base[f"sev{s}_p95_e"] = r.get("e_p95", float("nan"))
            else:
                base[f"sev{s}_DS"] = r.get("ds", float("nan"))
                base[f"sev{s}_BR"] = r.get("br", float("nan"))
                base[f"sev{s}_N_nb"] = r.get("n_nb", float("nan"))
                base[f"sev{s}_DS_nb"] = r.get("ds_nb", float("nan"))

        out.append(base)

    return out


def rq2_generalization_row(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in sorted(summary_rows, key=lambda x: (x.get("sim", ""), x.get("map", ""), x.get("model", ""))):
        sim = r.get("sim", "")
        if sim == "udacity":
            map_name = str(r.get("map", ""))
            is_jungle = map_name.lower() == "jungle"
            pr = float(r.get("mean_segments_passed_rate_per_run", np.nan)) if is_jungle else float(r.get("pass_rate", np.nan))
            out.append({
                "sim": "udacity",
                "map": map_name,
                "model": r.get("model", ""),
                "PR": pr,
                "E_dt": float(r.get("pid_dev_mean_pre_mean", np.nan)),
                "p95_dt": float(r.get("pid_dev_p95_pre_mean", np.nan)),
                "E_e": float(r.get("xte_abs_mean_pre_mean", np.nan)),
                "p95_e": float(r.get("xte_abs_p95_pre_mean", np.nan)),
            })
        else:
            if str(r.get("condition", "")).upper() != "ALL":
                continue
            out.append({
                "sim": "carla",
                "map": r.get("map", "multi-town"),
                "model": r.get("model", ""),
                "DS": float(r.get("ds_all_mean", np.nan)),
                "BR": float(r.get("blocked_rate", np.nan)),
                "N_nb": float(r.get("n_nb", np.nan)),
                "DS_nb": float(r.get("ds_nb_mean", np.nan)),
            })
    return out


def robustness_conditions_table(summary_rows: List[Dict[str, Any]], *, severities: Tuple[int, ...] = (2, 4)) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in summary_rows:
        if r.get("sim") != "udacity":
            continue
        
        pert_raw = r.get("perturbation", "")
        sev = int(r.get("severity", 0))
        is_base = _is_baseline(pert_raw)

        # -------------------------------------------------------------------
        # FIX: Check if it is baseline BEFORE checking severity match
        # -------------------------------------------------------------------
        if is_base:
            # Baseline is valid only if severity == 0 (which it usually is)
            if sev == 0:
                pert = "Baseline"
            else:
                continue
        else:
            # For non-baseline, we MUST match the requested severities (e.g. 2, 4)
            if sev not in severities:
                continue
            pert = str(pert_raw)

        map_name = str(r.get("map", ""))
        is_jungle = map_name.lower() == "jungle"
        pr = float(r.get("mean_segments_passed_rate_per_run", np.nan)) if is_jungle else float(r.get("pass_rate", np.nan))

        out.append({
            "sim": "udacity",
            "map": map_name,
            "model": r.get("model", ""),
            "perturbation": pert,
            "severity": sev,
            "PR": pr,
            "E_dt": float(r.get("pid_dev_mean_pre_mean", np.nan)),
            "p95_dt": float(r.get("pid_dev_p95_pre_mean", np.nan)),
            "E_e": float(r.get("xte_abs_mean_pre_mean", np.nan)),
            "p95_e": float(r.get("xte_abs_p95_pre_mean", np.nan)),
        })

    # Sort so Baseline comes first (False < True)
    return sorted(out, key=lambda x: (x["perturbation"] != "Baseline", x["perturbation"], x["severity"], x["map"], x["model"]))