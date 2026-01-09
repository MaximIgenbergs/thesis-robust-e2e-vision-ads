from __future__ import annotations
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence
import numpy as np

# ==============================================================================
# PART 1: IO + Helpers
# ==============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return [{k.strip(): v for k, v in row.items()} for row in r]

def write_csv_ordered(path: Path, rows: List[Dict[str, Any]], keys: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(keys), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_csv_multiheader(path: Path, header_rows: Sequence[Sequence[str]], data_rows: Sequence[Sequence[Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for hr in header_rows:
            w.writerow(list(hr))
        for dr in data_rows:
            w.writerow(list(dr))

def _find_files(root: Path, name: str) -> List[Path]:
    return sorted([p for p in root.rglob(name) if p.is_file()])

def _to_float_maybe(x: Any) -> Any:
    if x is None:
        return x
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    if s == "":
        return ""
    try:
        return float(s)
    except Exception:
        return s

def _norm(s: Any) -> str:
    return str(s).strip().lower().replace("-", "_")

def _load_all_rows_csv(path: Path) -> List[Dict[str, Any]]:
    rows = read_csv_rows(path)
    return [{k: _to_float_maybe(v) for k, v in r.items()} for r in rows]

def _safe_mean(xs: List[float]) -> float:
    if not xs:
        return -1.0
    arr = np.asarray(xs, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return -1.0
    return float(np.mean(valid))

def _mean_for_display(xs: List[float]) -> Any:
    v = _safe_mean(xs)
    return v if v != -1.0 else ""

def _calc_perf_score(pr: Any, dt: Any, e: Any) -> Tuple[float, float, float]:
    """Sort key: PR desc, dt asc, e asc -> return (PR, -dt, -e) for reverse sort."""
    def _f(v: Any) -> float:
        return float(v) if isinstance(v, (int, float)) else -1.0
    v_pr, v_dt, v_e = _f(pr), _f(dt), _f(e)
    final_dt = -v_dt if v_dt != -1.0 else float("-inf")
    final_e = -v_e if v_e != -1.0 else float("-inf")
    return (v_pr, final_dt, final_e)


# ==============================================================================
# PART 1A: Aggregated Tables (RQ1 Robustness)
# ==============================================================================
def build_table_rq1(results_root: Path, out_dir: Path) -> None:
    files_sev = _find_files(results_root, "robustness_by_severity.csv")
    unique_rows: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for fp in files_sev:
        for r in _load_all_rows_csv(fp):
            sim = _norm(r.get("sim", ""))
            if sim == "carla":
                continue
            map_name = _norm(r.get("map", ""))
            if "genroads" in map_name:
                display_map = "GenRoads"
            elif "jungle" in map_name:
                display_map = "Jungle"
            else:
                display_map = str(r.get("map", ""))
            sev = int(float(r.get("severity", -1)))
            if sev not in [0, 2, 4]:
                continue
            key = (sim, display_map, str(r.get("model", "")), sev)
            item = {
                "Sim": sim,
                "Map": display_map,
                "Model": r.get("model", ""),
                "Severity": sev,
                "PR": r.get("pr"),
                "E_dt": r.get("dt_mean"),
                "p95_dt": r.get("dt_p95"),
                "E_e": r.get("e_mean"),
                "p95_e": r.get("e_p95"),
                "DS": r.get("ds"),
                "BR": r.get("br"),
                "N_nb": r.get("n_nb"),
                "DS_nb": r.get("ds_nb"),
            }
            unique_rows[key] = item
    files_sum = _find_files(results_root, "summary_table.csv")
    carla_raw: List[Dict[str, Any]] = []
    for fp in files_sum:
        for r in _load_all_rows_csv(fp):
            if _norm(r.get("sim", "")) == "carla" and _norm(r.get("test_type", "")) == "robustness":
                carla_raw.append(r)
    carla_grouped = defaultdict(list)
    for r in carla_raw:
        sev = int(float(r.get("severity", -1)))
        if sev not in [0, 2, 4]:
            continue
        key = (str(r.get("model", "")), sev)
        carla_grouped[key].append(r)
    for (model, sev), items in carla_grouped.items():
        dss = [float(x["ds_all_mean"]) for x in items if x.get("ds_all_mean") not in [None, ""]]
        brs = [float(x["blocked_rate"]) for x in items if x.get("blocked_rate") not in [None, ""]]
        n_nbs = [float(x["n_nb"]) for x in items if x.get("n_nb") not in [None, ""]]
        ds_nbs = [float(x["ds_nb_mean"]) for x in items if x.get("ds_nb_mean") not in [None, ""]]
        item = {
            "Sim": "carla",
            "Map": "CARLA",
            "Model": model,
            "Severity": sev,
            "PR": None,
            "E_dt": None,
            "p95_dt": None,
            "E_e": None,
            "p95_e": None,
            "DS": _safe_mean(dss),
            "BR": _safe_mean(brs),
            "N_nb": _safe_mean(n_nbs),
            "DS_nb": _safe_mean(ds_nbs),
        }
        unique_rows[("carla", "CARLA", str(model), sev)] = item
    final_rows = list(unique_rows.values())
    if not final_rows:
        return
    final_rows.sort(key=lambda x: (x["Map"], str(x["Model"]), int(x["Severity"])))
    cols = ["Map", "Model", "Severity", "PR", "E_dt", "p95_dt", "E_e", "p95_e", "DS_nb", "BR", "N_nb", "DS"]
    write_csv_ordered(out_dir / "study_table_rq1_robustness.csv", final_rows, cols)


def build_table_rq2(results_root: Path, out_dir: Path) -> None:
    files = _find_files(results_root, "rq2_generalization_row.csv")
    raw_rows: List[Dict[str, Any]] = []
    for fp in files:
        raw_rows.extend(_load_all_rows_csv(fp))
    metric_keys = ["PR", "E_dt", "p95_dt", "E_e", "p95_e", "DS", "BR", "N_nb", "DS_nb"]
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in raw_rows:
        sim = _norm(r.get("sim", ""))
        raw_map = _norm(r.get("map", ""))
        if sim == "carla":
            display_map = "CARLA"
        elif "genroads" in raw_map:
            display_map = "GenRoads"
        elif "jungle" in raw_map:
            display_map = "Jungle"
        else:
            display_map = str(r.get("map", ""))
        key = (sim, display_map, str(r.get("model", "")))
        groups[key].append(r)
    
    # Load tiny routes data for CARLA
    tiny_routes_data = {}
    tiny_routes_path = Path("/media/maxim/Elements/maximigenbergs/runs/carla/generalization/tcp/20251231_045136/tcp")
    towns = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06"]
    
    if tiny_routes_path.exists():
        all_tiny_routes = []
        for town in towns:
            # Use same method for all towns
            town_path = tiny_routes_path / town
            if town_path.exists():
                json_files = list(town_path.rglob("simulation_results*.json"))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        from metrics.io_carla import parse_carla_routes, route_driving_score, is_blocked_route
                        routes = parse_carla_routes(data)
                        
                        for rec in routes:
                            ds = route_driving_score(rec)
                            if ds is not None:
                                all_tiny_routes.append({
                                    "driving_score": float(ds),
                                    "blocked": int(is_blocked_route(rec))
                                })
                    except Exception as e:
                        print(f"Warning: Could not process {json_file}: {e}")
                        continue
        
        if all_tiny_routes:
            dss = [r["driving_score"] for r in all_tiny_routes]
            blocked = [r["blocked"] for r in all_tiny_routes]
            br = sum(blocked) / len(blocked) if blocked else 0
            n_nb = sum(1 for b in blocked if b == 0)
            ds_nb_vals = [r["driving_score"] for r in all_tiny_routes if r["blocked"] == 0]
            
            tiny_routes_data = {
                "DS": _mean_for_display(dss),
                "BR": br,
                "N_nb": n_nb,
                "DS_nb": _mean_for_display(ds_nb_vals),
            }
    
    aggregated_rows: List[Dict[str, Any]] = []
    for (sim, map_name, model), items in groups.items():
        base: Dict[str, Any] = {"sim": sim, "map": map_name, "model": model}
        for m in metric_keys:
            vals: List[float] = []
            for it in items:
                v = it.get(m)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            base[m] = _mean_for_display(vals)
        
        # Add tiny routes data for CARLA
        if sim == "carla" and tiny_routes_data:
            # Add tiny routes metrics with "_tiny" suffix
            for metric in ["DS", "BR", "N_nb", "DS_nb"]:
                base[f"{metric}_tiny"] = tiny_routes_data.get(metric, "")
        
        aggregated_rows.append(base)
    
    def map_grouper(r: Dict[str, Any]) -> int:
        m_name = _norm(r.get("map", ""))
        if "genroads" in m_name:
            return 0
        if "jungle" in m_name:
            return 1
        return 2
    aggregated_rows.sort(key=lambda x: (map_grouper(x), str(x.get("model", ""))))
    
    # Update columns to include tiny routes metrics for CARLA (reordered)
    cols = ["sim", "map", "model", "PR", "E_dt", "p95_dt", "E_e", "p95_e", "DS_nb", "BR", "N_nb", "DS"]
    # Add tiny routes columns (they'll be empty for non-CARLA rows) (reordered)
    cols.extend(["DS_nb_tiny", "BR_tiny", "N_nb_tiny", "DS_tiny"])
    
    write_csv_ordered(out_dir / "study_table_rq2_generalization.csv", aggregated_rows, cols)


# ==============================================================================
# PART 1B: Per-Perturbation Pivot Tables (RQ1 Robustness)
# ==============================================================================
def build_pivot_perturbations_independent(results_root: Path, out_dir: Path) -> None:
    # 1) Udacity: keep ONLY 3 metrics per model to match LaTeX renderer
    files_udacity = _find_files(results_root, "robustness_by_perturbation.csv")
    rows_udacity: List[Dict[str, Any]] = []
    for fp in files_udacity:
        rows_udacity.extend(_load_all_rows_csv(fp))
    udacity_rows = [r for r in rows_udacity if _norm(r.get("sim", "")) == "udacity"]
    if udacity_rows:
        maps = ["genroads", "jungle"]
        models = ["dave2", "dave2_gru", "vit"]
        idx: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
        keys_per_map: Dict[str, set] = defaultdict(set)
        for r in udacity_rows:
            p_raw = str(r.get("perturbation", "")).strip()
            sev_raw = r.get("severity", "")
            try:
                sev = int(float(sev_raw))
            except Exception:
                sev = 0
            if p_raw.lower() in ["baseline", "clean", "none"] or sev == 0:
                pert, sev = "Baseline", 0
            else:
                pert = p_raw
            m = _norm(r.get("map", ""))
            mdl = _norm(r.get("model", ""))
            idx[(pert, sev, m, mdl)] = r
            keys_per_map[m].add((pert, sev))
        for target_map in maps:
            def score_key(k: Tuple[str, int]) -> Tuple[float, float, float]:
                pert, sev = k
                if pert == "Baseline":
                    return (float("inf"), float("inf"), float("inf"))
                prs: List[float] = []
                dts: List[float] = []
                es: List[float] = []
                for mdl in models:
                    rr = idx.get((pert, sev, target_map, mdl), {})
                    v_pr = rr.get("PR")
                    v_dt = rr.get("E_dt")
                    v_e = rr.get("E_e")
                    if isinstance(v_pr, (int, float)):
                        prs.append(float(v_pr))
                    if isinstance(v_dt, (int, float)):
                        dts.append(float(v_dt))
                    if isinstance(v_e, (int, float)):
                        es.append(float(v_e))
                return _calc_perf_score(_safe_mean(prs), _safe_mean(dts), _safe_mean(es))
            map_keys = sorted(list(keys_per_map[target_map]), key=score_key, reverse=True)
            metrics = ["PR", "E_dt", "E_e"]
            h1 = ["", ""]
            h1.extend([target_map] + [""] * (len(models) * len(metrics) - 1))
            h2 = ["", ""]
            for mdl in models:
                h2.extend([mdl] + [""] * (len(metrics) - 1))
            h3 = ["perturbation", "severity"] + metrics * len(models)
            data_rows: List[List[Any]] = []
            for (pert, sev) in map_keys:
                row: List[Any] = [pert, sev]
                for mdl in models:
                    for met in metrics:
                        rr = idx.get((pert, sev, target_map, mdl), {})
                        row.append(rr.get(met, ""))
                data_rows.append(row)
            fname = f"study_table_rq1_robustness_by_perturbation_{target_map}.csv"
            write_csv_multiheader(out_dir / fname, header_rows=[h1, h2, h3], data_rows=data_rows)
    # 2) CARLA pivot (robust to multiple rows by averaging)
    files_summary = _find_files(results_root, "summary_table.csv")
    carla_raw: List[Dict[str, Any]] = []
    for fp in files_summary:
        for r in _load_all_rows_csv(fp):
            if _norm(r.get("sim", "")) == "carla" and _norm(r.get("test_type", "")) == "robustness":
                carla_raw.append(r)
    if carla_raw:
        grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for r in carla_raw:
            p_raw = str(r.get("perturbation", "")).strip()
            sev_raw = r.get("severity", "")
            try:
                sev = int(float(sev_raw))
            except Exception:
                sev = 0
            if p_raw.lower() in ["baseline", "clean", "none"] or sev == 0:
                pert, sev = "Baseline", 0
            else:
                pert = p_raw
            grouped[(pert, sev)].append(r)
        def agg_row(items: List[Dict[str, Any]]) -> Tuple[Any, Any, Any, Any]:
            dsnb = [float(x["ds_nb_mean"]) for x in items if x.get("ds_nb_mean") not in [None, ""]]
            brs = [float(x["blocked_rate"]) for x in items if x.get("blocked_rate") not in [None, ""]]
            nbs = [float(x["n_nb"]) for x in items if x.get("n_nb") not in [None, ""]]
            dss = [float(x["ds_all_mean"]) for x in items if x.get("ds_all_mean") not in [None, ""]]
            return (_mean_for_display(dsnb), _mean_for_display(brs), _mean_for_display(nbs), _mean_for_display(dss))
        def sort_key(k: Tuple[str, int]) -> Tuple[float, float, float]:
            pert, sev = k
            if pert == "Baseline":
                return (float("inf"), float("inf"), float("inf"))
            dsnb, br, _, ds = agg_row(grouped[k])
            dsnb_f = float(dsnb) if isinstance(dsnb, (int, float)) else float("-inf")
            br_f = float(br) if isinstance(br, (int, float)) else float("inf")
            ds_f = float(ds) if isinstance(ds, (int, float)) else float("-inf")
            return (dsnb_f, -br_f, ds_f)
        sorted_keys = sorted(list(grouped.keys()), key=sort_key, reverse=True)
        h1 = ["CARLA"] + [""] * 5
        h2 = ["perturbation", "severity", "DS_nb", "BR", "N_nb", "DS"]
        d_rows: List[List[Any]] = []
        for (pert, sev) in sorted_keys:
            dsnb, br, nb, ds = agg_row(grouped[(pert, sev)])
            d_rows.append([pert, sev, dsnb, br, nb, ds])
        write_csv_multiheader(out_dir / "study_table_rq1_robustness_by_perturbation_carla.csv", header_rows=[h1, h2], data_rows=d_rows)


# ==============================================================================
# PART 1C: NEW Generalization Detail Tables (RQ2)
# ==============================================================================

def build_table_rq2_genroads_by_scenario(results_root: Path, out_dir: Path) -> None:
    """
    GenRoads Generalization Table by Scenario Type:
    - Rows: scenario types (e.g., chicane_lr, hairpin_l, wide_hairpin_r, etc.)
    - Columns: 3 models (dave2, dave2_gru, vit) × 3 metrics (PR, E_dt, E_e)
    - Aggregation: For each (scenario_type, model) combination, average over ALL episodes of that scenario type
    - Sorted by performance (PR desc, dt asc, e asc)
    """
    # Find entry_metrics.csv files for genroads generalization
    files = _find_files(results_root, "entry_metrics.csv")
    
    all_entries: List[Dict[str, Any]] = []
    for fp in files:
        # Only include genroads generalization data
        path_str = str(fp).lower()
        if "genroads" in path_str and "generalization" in path_str:
            all_entries.extend(_load_all_rows_csv(fp))
    
    if not all_entries:
        print("WARNING: No GenRoads generalization entry_metrics.csv found")
        return
    
    models = ["dave2", "dave2_gru", "vit"]
    
    # Group entries by (scenario_type, model) - using task_id as scenario type
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    scenarios_seen: set = set()
    
    for entry in all_entries:
        model = _norm(entry.get("model", ""))
        if model not in models:
            continue
        
        # Use 'task_id' field as the scenario type
        scenario_type = str(entry.get("task_id", "unknown"))
        scenarios_seen.add(scenario_type)
        grouped[(scenario_type, model)].append(entry)
    
    if not scenarios_seen:
        print("WARNING: No scenario types found in GenRoads generalization data")
        return
    
    # Build aggregated data
    # idx[(scenario_type, model)] = {PR: ..., E_dt: ..., E_e: ...}
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    for (scenario_type, model), entries in grouped.items():
        # Aggregate metrics
        prs = [float(e["is_success"]) for e in entries if e.get("is_success") is not None]
        dts = [float(e["pid_dev_mean_pre"]) for e in entries if e.get("pid_dev_mean_pre") not in [None, ""]]
        es = [float(e["xte_abs_mean_pre"]) for e in entries if e.get("xte_abs_mean_pre") not in [None, ""]]
        
        idx[(scenario_type, model)] = {
            "PR": _mean_for_display(prs),
            "E_dt": _mean_for_display(dts),
            "E_e": _mean_for_display(es),
        }
    
    # Calculate sort scores for each scenario type (average across models)
    scenario_scores: Dict[str, Tuple[float, float, float]] = {}
    for scenario_type in scenarios_seen:
        prs_for_score: List[float] = []
        dts_for_score: List[float] = []
        es_for_score: List[float] = []
        for mdl in models:
            data = idx.get((scenario_type, mdl), {})
            v_pr = data.get("PR")
            v_dt = data.get("E_dt")
            v_e = data.get("E_e")
            if isinstance(v_pr, (int, float)):
                prs_for_score.append(float(v_pr))
            if isinstance(v_dt, (int, float)):
                dts_for_score.append(float(v_dt))
            if isinstance(v_e, (int, float)):
                es_for_score.append(float(v_e))
        scenario_scores[scenario_type] = _calc_perf_score(
            _safe_mean(prs_for_score),
            _safe_mean(dts_for_score),
            _safe_mean(es_for_score)
        )
    
    # Sort scenario types by performance (descending)
    sorted_scenarios = sorted(list(scenarios_seen), key=lambda s: scenario_scores.get(s, (-1, float("-inf"), float("-inf"))), reverse=True)
    
    # Build CSV with multi-header
    metrics = ["PR", "E_dt", "E_e"]
    
    # Header row 1: genroads spanning all model columns
    h1 = [""]
    h1.extend(["genroads"] + [""] * (len(models) * len(metrics) - 1))
    
    # Header row 2: model names
    h2 = [""]
    for mdl in models:
        h2.extend([mdl] + [""] * (len(metrics) - 1))
    
    # Header row 3: metric names
    h3 = ["scenario"] + metrics * len(models)
    
    # Build data rows (sorted by performance)
    data_rows: List[List[Any]] = []
    for scenario_type in sorted_scenarios:
        row: List[Any] = [scenario_type]
        for mdl in models:
            data = idx.get((scenario_type, mdl), {})
            for met in metrics:
                row.append(data.get(met, ""))
        data_rows.append(row)
    
    fname = "study_table_rq2_generalization_genroads_by_scenario.csv"
    write_csv_multiheader(out_dir / fname, header_rows=[h1, h2, h3], data_rows=data_rows)
    print(f"[OK] Built GenRoads generalization by scenario table: {len(data_rows)} rows (scenario types)")


def build_table_rq2_genroads_detail(results_root: Path, out_dir: Path) -> None:
    """
    GenRoads Generalization Table:
    - Rows: roads (e.g., wide_hairpin_r, etc.)
    - Columns: 3 models (dave2, dave2_gru, vit) × 3 metrics (PR, E_dt, E_e)
    - Aggregation: For each (road, model) combination, average over ALL scenarios on that road
    - Sorted by performance (PR desc, dt asc, e asc)
    """
    # Find entry_metrics.csv files for genroads generalization
    files = _find_files(results_root, "entry_metrics.csv")
    
    all_entries: List[Dict[str, Any]] = []
    for fp in files:
        # Only include genroads generalization data
        path_str = str(fp).lower()
        if "genroads" in path_str and "generalization" in path_str:
            all_entries.extend(_load_all_rows_csv(fp))
    
    if not all_entries:
        print("WARNING: No GenRoads generalization entry_metrics.csv found")
        return
    
    models = ["dave2", "dave2_gru", "vit"]
    
    # Group entries by (road, model)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    roads_seen: set = set()
    
    for entry in all_entries:
        model = _norm(entry.get("model", ""))
        if model not in models:
            continue
        
        # Use 'road' field as the grouping key
        road = str(entry.get("road", "unknown"))
        roads_seen.add(road)
        grouped[(road, model)].append(entry)
    
    if not roads_seen:
        print("WARNING: No roads found in GenRoads generalization data")
        return
    
    # Build aggregated data
    # idx[(road, model)] = {PR: ..., E_dt: ..., E_e: ...}
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    for (road, model), entries in grouped.items():
        # Aggregate metrics
        prs = [float(e["is_success"]) for e in entries if e.get("is_success") is not None]
        dts = [float(e["pid_dev_mean_pre"]) for e in entries if e.get("pid_dev_mean_pre") not in [None, ""]]
        es = [float(e["xte_abs_mean_pre"]) for e in entries if e.get("xte_abs_mean_pre") not in [None, ""]]
        
        idx[(road, model)] = {
            "PR": _mean_for_display(prs),
            "E_dt": _mean_for_display(dts),
            "E_e": _mean_for_display(es),
        }
    
    # Calculate sort scores for each road (average across models)
    road_scores: Dict[str, Tuple[float, float, float]] = {}
    for road in roads_seen:
        prs_for_score: List[float] = []
        dts_for_score: List[float] = []
        es_for_score: List[float] = []
        for mdl in models:
            data = idx.get((road, mdl), {})
            v_pr = data.get("PR")
            v_dt = data.get("E_dt")
            v_e = data.get("E_e")
            if isinstance(v_pr, (int, float)):
                prs_for_score.append(float(v_pr))
            if isinstance(v_dt, (int, float)):
                dts_for_score.append(float(v_dt))
            if isinstance(v_e, (int, float)):
                es_for_score.append(float(v_e))
        road_scores[road] = _calc_perf_score(
            _safe_mean(prs_for_score),
            _safe_mean(dts_for_score),
            _safe_mean(es_for_score)
        )
    
    # Sort roads by performance (descending)
    sorted_roads = sorted(list(roads_seen), key=lambda r: road_scores.get(r, (-1, float("-inf"), float("-inf"))), reverse=True)
    
    # Build CSV with multi-header
    metrics = ["PR", "E_dt", "E_e"]
    
    # Header row 1: genroads spanning all model columns
    h1 = [""]
    h1.extend(["genroads"] + [""] * (len(models) * len(metrics) - 1))
    
    # Header row 2: model names
    h2 = [""]
    for mdl in models:
        h2.extend([mdl] + [""] * (len(metrics) - 1))
    
    # Header row 3: metric names
    h3 = ["road"] + metrics * len(models)
    
    # Build data rows (sorted by performance)
    data_rows: List[List[Any]] = []
    for road in sorted_roads:
        row: List[Any] = [road]
        for mdl in models:
            data = idx.get((road, mdl), {})
            for met in metrics:
                row.append(data.get(met, ""))
        data_rows.append(row)
    
    fname = "study_table_rq2_generalization_genroads_detail.csv"
    write_csv_multiheader(out_dir / fname, header_rows=[h1, h2, h3], data_rows=data_rows)
    print(f"[OK] Built GenRoads generalization detail table: {len(data_rows)} rows (roads)")


def build_table_rq2_genroads_scenarios(results_root: Path, out_dir: Path) -> None:
    """
    GenRoads Generalization Table by Scenario Types:
    - Rows: scenario types (baseline, btap, sbias, sfreeze, sgain, spulse, tboost)
    - Columns: 3 models (dave2, dave2_gru, vit) × 3 metrics (PR, E_dt, E_e)
    - Aggregation: For each (scenario_type, model) combination, average over ALL perturbations and roads in that scenario type
    - Sorted by performance (PR desc, dt asc, e asc)
    """
    # Find entry_metrics.csv files for genroads generalization
    files = _find_files(results_root, "entry_metrics.csv")
    
    all_entries: List[Dict[str, Any]] = []
    for fp in files:
        # Only include genroads generalization data
        path_str = str(fp).lower()
        if "genroads" in path_str and "generalization" in path_str:
            all_entries.extend(_load_all_rows_csv(fp))
    
    if not all_entries:
        print("WARNING: No GenRoads generalization entry_metrics.csv found")
        return
    
    models = ["dave2", "dave2_gru", "vit"]
    
    # Define scenario type mapping based on perturbation prefixes
    def get_scenario_type(perturbation: str) -> str:
        pert = str(perturbation).lower().strip()
        if pert == "baseline":
            return "baseline"
        elif pert.startswith("btap"):
            return "btap"
        elif pert.startswith("sbias"):
            return "sbias"
        elif pert.startswith("sfreeze"):
            return "sfreeze"
        elif pert.startswith("sgain"):
            return "sgain"
        elif pert.startswith("spulse"):
            return "spulse"
        elif pert.startswith("tboost"):
            return "tboost"
        else:
            return "unknown"
    
    # Group entries by (scenario_type, model)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    scenario_types_seen: set = set()
    
    for entry in all_entries:
        model = _norm(entry.get("model", ""))
        if model not in models:
            continue
        
        perturbation = str(entry.get("perturbation", ""))
        scenario_type = get_scenario_type(perturbation)
        if scenario_type == "unknown":
            continue
            
        scenario_types_seen.add(scenario_type)
        grouped[(scenario_type, model)].append(entry)
    
    if not scenario_types_seen:
        print("WARNING: No scenario types found in GenRoads generalization data")
        return
    
    # Build aggregated data
    # idx[(scenario_type, model)] = {PR: ..., E_dt: ..., E_e: ...}
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    for (scenario_type, model), entries in grouped.items():
        # Aggregate metrics
        prs = [float(e["is_success"]) for e in entries if e.get("is_success") is not None]
        dts = [float(e["pid_dev_mean_pre"]) for e in entries if e.get("pid_dev_mean_pre") not in [None, ""]]
        es = [float(e["xte_abs_mean_pre"]) for e in entries if e.get("xte_abs_mean_pre") not in [None, ""]]
        
        idx[(scenario_type, model)] = {
            "PR": _mean_for_display(prs),
            "E_dt": _mean_for_display(dts),
            "E_e": _mean_for_display(es),
        }
    
    # Calculate sort scores for each scenario type (average across models)
    scenario_scores: Dict[str, Tuple[float, float, float]] = {}
    for scenario_type in scenario_types_seen:
        prs_for_score: List[float] = []
        dts_for_score: List[float] = []
        es_for_score: List[float] = []
        for mdl in models:
            data = idx.get((scenario_type, mdl), {})
            v_pr = data.get("PR")
            v_dt = data.get("E_dt")
            v_e = data.get("E_e")
            if isinstance(v_pr, (int, float)):
                prs_for_score.append(float(v_pr))
            if isinstance(v_dt, (int, float)):
                dts_for_score.append(float(v_dt))
            if isinstance(v_e, (int, float)):
                es_for_score.append(float(v_e))
        scenario_scores[scenario_type] = _calc_perf_score(
            _safe_mean(prs_for_score),
            _safe_mean(dts_for_score),
            _safe_mean(es_for_score)
        )
    
    # Sort scenario types by performance (descending)
    sorted_scenarios = sorted(list(scenario_types_seen), key=lambda s: scenario_scores.get(s, (-1, float("-inf"), float("-inf"))), reverse=True)
    
    # Build CSV with multi-header
    metrics = ["PR", "E_dt", "E_e"]
    
    # Header row 1: genroads spanning all model columns
    h1 = [""]
    h1.extend(["genroads"] + [""] * (len(models) * len(metrics) - 1))
    
    # Header row 2: model names
    h2 = [""]
    for mdl in models:
        h2.extend([mdl] + [""] * (len(metrics) - 1))
    
    # Header row 3: metric names
    h3 = ["scenario"] + metrics * len(models)
    
    # Build data rows (sorted by performance)
    data_rows: List[List[Any]] = []
    for scenario_type in sorted_scenarios:
        row: List[Any] = [scenario_type]
        for mdl in models:
            data = idx.get((scenario_type, mdl), {})
            for met in metrics:
                row.append(data.get(met, ""))
        data_rows.append(row)
    
    fname = "study_table_rq2_generalization_genroads_scenarios.csv"
    write_csv_multiheader(out_dir / fname, header_rows=[h1, h2, h3], data_rows=data_rows)
    print(f"[OK] Built GenRoads generalization scenarios table: {len(data_rows)} rows (scenario types)")


def build_table_rq2_jungle_detail(results_root: Path, out_dir: Path) -> None:
    """
    Jungle Generalization Table:
    - Rows: 13 segments (task_id/segment names)
    - Columns: 3 models × 3 metrics (PR, E_dt, E_e)
    - Aggregation: Average over all episodes (runs) per segment per model
    - Sorted by performance (PR desc, dt asc, e asc)
    """
    # Find entry_metrics.csv files for jungle generalization
    files = _find_files(results_root, "entry_metrics.csv")
    
    all_entries: List[Dict[str, Any]] = []
    for fp in files:
        path_str = str(fp).lower()
        if "jungle" in path_str and "generalization" in path_str:
            all_entries.extend(_load_all_rows_csv(fp))
    
    if not all_entries:
        print("WARNING: No Jungle generalization entry_metrics.csv found")
        return
    
    models = ["dave2", "dave2_gru", "vit"]
    
    # Group entries by (segment/task_id, model)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    segments_seen: set = set()
    
    for entry in all_entries:
        model = _norm(entry.get("model", ""))
        if model not in models:
            continue
        
        segment = str(entry.get("task_id", "unknown"))
        segments_seen.add(segment)
        grouped[(segment, model)].append(entry)
    
    if not segments_seen:
        print("WARNING: No segments found in Jungle generalization data")
        return
    
    # Build aggregated data
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    for (segment, model), entries in grouped.items():
        prs = [float(e["is_success"]) for e in entries if e.get("is_success") is not None]
        dts = [float(e["pid_dev_mean_pre"]) for e in entries if e.get("pid_dev_mean_pre") not in [None, ""]]
        es = [float(e["xte_abs_mean_pre"]) for e in entries if e.get("xte_abs_mean_pre") not in [None, ""]]
        
        idx[(segment, model)] = {
            "PR": _mean_for_display(prs),
            "E_dt": _mean_for_display(dts),
            "E_e": _mean_for_display(es),
        }
    
    # Calculate sort scores for each segment (average across models)
    segment_scores: Dict[str, Tuple[float, float, float]] = {}
    for segment in segments_seen:
        prs_for_score: List[float] = []
        dts_for_score: List[float] = []
        es_for_score: List[float] = []
        for mdl in models:
            data = idx.get((segment, mdl), {})
            v_pr = data.get("PR")
            v_dt = data.get("E_dt")
            v_e = data.get("E_e")
            if isinstance(v_pr, (int, float)):
                prs_for_score.append(float(v_pr))
            if isinstance(v_dt, (int, float)):
                dts_for_score.append(float(v_dt))
            if isinstance(v_e, (int, float)):
                es_for_score.append(float(v_e))
        segment_scores[segment] = _calc_perf_score(
            _safe_mean(prs_for_score),
            _safe_mean(dts_for_score),
            _safe_mean(es_for_score)
        )
    
    # Sort segments by performance (descending)
    sorted_segments = sorted(list(segments_seen), key=lambda s: segment_scores.get(s, (-1, float("-inf"), float("-inf"))), reverse=True)
    
    # Build CSV with multi-header
    metrics = ["PR", "E_dt", "E_e"]
    
    # Header row 1: jungle spanning all model columns
    h1 = [""]
    h1.extend(["jungle"] + [""] * (len(models) * len(metrics) - 1))
    
    # Header row 2: model names
    h2 = [""]
    for mdl in models:
        h2.extend([mdl] + [""] * (len(metrics) - 1))
    
    # Header row 3: metric names
    h3 = ["segment"] + metrics * len(models)
    
    # Build data rows (sorted by performance)
    data_rows: List[List[Any]] = []
    
    for segment in sorted_segments:
        row: List[Any] = [segment]
        for mdl in models:
            data = idx.get((segment, mdl), {})
            for met in metrics:
                row.append(data.get(met, ""))
        data_rows.append(row)
    
    fname = "study_table_rq2_generalization_jungle_detail.csv"
    write_csv_multiheader(out_dir / fname, header_rows=[h1, h2, h3], data_rows=data_rows)
    print(f"[OK] Built Jungle generalization detail table: {len(data_rows)} rows (segments)")


def build_table_rq1_jungle_detail(results_root: Path, out_dir: Path) -> None:
    """
    Jungle Robustness Table:
    - Rows: segments (task_id/segment names)
    - Columns: 3 models × 3 metrics (PR, E_dt, E_e)
    - Aggregation: Average over all episodes (runs) per segment per model
    - Sorted by performance (PR desc, dt asc, e asc)
    """
    # Find entry_metrics.csv files for jungle robustness
    files = _find_files(results_root, "entry_metrics.csv")
    
    all_entries: List[Dict[str, Any]] = []
    for fp in files:
        path_str = str(fp).lower()
        if "jungle" in path_str and "robustness" in path_str:
            all_entries.extend(_load_all_rows_csv(fp))
    
    if not all_entries:
        print("WARNING: No Jungle robustness entry_metrics.csv found")
        return
    
    models = ["dave2", "dave2_gru", "vit"]
    
    # Group entries by (segment/task_id, model)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    segments_seen: set = set()
    
    for entry in all_entries:
        model = _norm(entry.get("model", ""))
        if model not in models:
            continue
        
        segment = str(entry.get("task_id", "unknown"))
        segments_seen.add(segment)
        grouped[(segment, model)].append(entry)
    
    if not segments_seen:
        print("WARNING: No segments found in Jungle robustness data")
        return
    
    # Build aggregated data
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    for (segment, model), entries in grouped.items():
        prs = [float(e["is_success"]) for e in entries if e.get("is_success") is not None]
        dts = [float(e["pid_dev_mean_pre"]) for e in entries if e.get("pid_dev_mean_pre") not in [None, ""]]
        es = [float(e["xte_abs_mean_pre"]) for e in entries if e.get("xte_abs_mean_pre") not in [None, ""]]
        
        idx[(segment, model)] = {
            "PR": _mean_for_display(prs),
            "E_dt": _mean_for_display(dts),
            "E_e": _mean_for_display(es),
        }
    
    # Calculate sort scores for each segment (average across models)
    segment_scores: Dict[str, Tuple[float, float, float]] = {}
    for segment in segments_seen:
        prs_for_score: List[float] = []
        dts_for_score: List[float] = []
        es_for_score: List[float] = []
        for mdl in models:
            data = idx.get((segment, mdl), {})
            v_pr = data.get("PR")
            v_dt = data.get("E_dt")
            v_e = data.get("E_e")
            if isinstance(v_pr, (int, float)):
                prs_for_score.append(float(v_pr))
            if isinstance(v_dt, (int, float)):
                dts_for_score.append(float(v_dt))
            if isinstance(v_e, (int, float)):
                es_for_score.append(float(v_e))
        segment_scores[segment] = _calc_perf_score(
            _safe_mean(prs_for_score),
            _safe_mean(dts_for_score),
            _safe_mean(es_for_score)
        )
    
    # Sort segments by performance (descending)
    sorted_segments = sorted(list(segments_seen), key=lambda s: segment_scores.get(s, (-1, float("-inf"), float("-inf"))), reverse=True)
    
    # Build CSV with multi-header
    metrics = ["PR", "E_dt", "E_e"]
    
    # Header row 1: jungle spanning all model columns
    h1 = [""]
    h1.extend(["jungle"] + [""] * (len(models) * len(metrics) - 1))
    
    # Header row 2: model names
    h2 = [""]
    for mdl in models:
        h2.extend([mdl] + [""] * (len(metrics) - 1))
    
    # Header row 3: metric names
    h3 = ["segment"] + metrics * len(models)
    
    # Build data rows (sorted by performance)
    data_rows: List[List[Any]] = []
    
    for segment in sorted_segments:
        row: List[Any] = [segment]
        for mdl in models:
            data = idx.get((segment, mdl), {})
            for met in metrics:
                row.append(data.get(met, ""))
        data_rows.append(row)
    
    fname = "study_table_rq1_robustness_jungle_detail.csv"
    write_csv_multiheader(out_dir / fname, header_rows=[h1, h2, h3], data_rows=data_rows)
    print(f"[OK] Built Jungle robustness detail table: {len(data_rows)} rows (segments)")


def build_table_rq2_carla_detail(results_root: Path, out_dir: Path) -> None:
    """
    CARLA Generalization Table:
    - Rows: Towns 1-6
    - Columns: Long Routes (DS_nb, BR, N_nb, DS) | Tiny Routes (DS_nb, BR, N_nb, DS)
    - Long Routes: real data from generalization runs
    - Tiny Routes: data from /media/maxim/Elements/maximigenbergs/runs/carla/generalization/tcp/20251231_045136/tcp/
    - Sorted by performance (DS_nb desc, BR asc, DS desc)
    """
    # Find route_metrics.csv files for carla generalization
    files = _find_files(results_root, "route_metrics.csv")
    
    all_routes: List[Dict[str, Any]] = []
    for fp in files:
        path_str = str(fp).lower()
        if "carla" in path_str and "generalization" in path_str:
            all_routes.extend(_load_all_rows_csv(fp))
    
    # Group by town (extract town from condition field, e.g., "Town01_01" -> "Town01")
    towns = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06"]
    
    long_routes_by_town: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for route in all_routes:
        condition = str(route.get("condition", ""))
        for town in towns:
            if town.lower() in condition.lower():
                long_routes_by_town[town].append(route)
                break
    
    # Build aggregated data for long routes
    long_data: Dict[str, Dict[str, Any]] = {}
    
    for town in towns:
        # Use same method for all towns
        routes = long_routes_by_town.get(town, [])
        if routes:
            dss = [float(r["driving_score"]) for r in routes if r.get("driving_score") not in [None, ""]]
            blocked = [int(r.get("blocked", 0)) for r in routes]
            br = sum(blocked) / len(blocked) if blocked else 0
            n_nb = sum(1 for b in blocked if b == 0)
            ds_nb_vals = [float(r["driving_score"]) for r, b in zip(routes, blocked) 
                          if b == 0 and r.get("driving_score") not in [None, ""]]
            
            long_data[town] = {
                "DS": _mean_for_display(dss),
                "BR": br if routes else "",
                "N_nb": n_nb if routes else "",
                "DS_nb": _mean_for_display(ds_nb_vals),
            }
        else:
            long_data[town] = {"DS": "", "BR": "", "N_nb": "", "DS_nb": ""}
    
    # Tiny routes: load data from the specified path
    tiny_routes_path = Path("/media/maxim/Elements/maximigenbergs/runs/carla/generalization/tcp/20251231_045136/tcp")
    tiny_data: Dict[str, Dict[str, Any]] = {}
    
    for town in towns:
        # Use same method for all towns
        town_path = tiny_routes_path / town
        if town_path.exists():
            # Find simulation_results*.json files in this town folder
            json_files = list(town_path.rglob("simulation_results*.json"))
            
            if json_files:
                # Process the tiny routes data using the same logic as long routes
                tiny_routes = []
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Parse routes using existing function
                        from metrics.io_carla import parse_carla_routes, route_driving_score, is_blocked_route
                        routes = parse_carla_routes(data)
                        
                        for rec in routes:
                            ds = route_driving_score(rec)
                            if ds is not None:
                                tiny_routes.append({
                                    "driving_score": float(ds),
                                    "blocked": int(is_blocked_route(rec))
                                })
                    except Exception as e:
                        print(f"Warning: Could not process {json_file}: {e}")
                        continue
                
                if tiny_routes:
                    dss = [r["driving_score"] for r in tiny_routes]
                    blocked = [r["blocked"] for r in tiny_routes]
                    br = sum(blocked) / len(blocked) if blocked else 0
                    n_nb = sum(1 for b in blocked if b == 0)
                    ds_nb_vals = [r["driving_score"] for r in tiny_routes if r["blocked"] == 0]
                    
                    tiny_data[town] = {
                        "DS": _mean_for_display(dss),
                        "BR": br,
                        "N_nb": n_nb,
                        "DS_nb": _mean_for_display(ds_nb_vals),
                    }
                else:
                    tiny_data[town] = {"DS": "", "BR": "", "N_nb": "", "DS_nb": ""}
            else:
                tiny_data[town] = {"DS": "", "BR": "", "N_nb": "", "DS_nb": ""}
        else:
            # Town folder doesn't exist, use empty values
            tiny_data[town] = {"DS": "", "BR": "", "N_nb": "", "DS_nb": ""}
    
    # Calculate sort scores for each town (based on long routes performance)
    def _calc_carla_score(ds_nb: Any, br: Any, ds: Any) -> Tuple[float, float, float]:
        def _f(v: Any) -> float:
            return float(v) if isinstance(v, (int, float)) and v != "" else float("-inf")
        v_ds_nb = _f(ds_nb)
        v_br = _f(br)
        v_ds = _f(ds)
        # For BR, lower is better, so negate it
        final_br = -v_br if v_br != float("-inf") else float("inf")
        return (v_ds_nb, final_br, v_ds)
    
    town_scores: Dict[str, Tuple[float, float, float]] = {}
    for town in towns:
        data = long_data.get(town, {})
        v_ds_nb = data.get("DS_nb")
        v_br = data.get("BR")
        v_ds = data.get("DS")
        town_scores[town] = _calc_carla_score(v_ds_nb, v_br, v_ds)
    
    # Sort towns by performance (descending)
    sorted_towns = sorted(towns, key=lambda t: town_scores.get(t, (float("-inf"), float("inf"), float("-inf"))), reverse=True)
    
    # Build CSV with multi-header
    metrics = ["DS_nb", "BR", "N_nb", "DS"]  # Reordered: DS_nb first, DS last
    
    # Header row 1: Long Routes | Tiny Routes
    h1 = [""]
    h1.extend(["Long Routes"] + [""] * (len(metrics) - 1))
    h1.extend(["Tiny Routes"] + [""] * (len(metrics) - 1))
    
    # Header row 2: metric names (repeated for each route type)
    h2 = ["Town"] + metrics + metrics
    
    # Build data rows (sorted by performance)
    data_rows: List[List[Any]] = []
    
    for town in sorted_towns:
        row: List[Any] = [town]
        # Long routes metrics (reordered: DS_nb, BR, N_nb, DS)
        for met in metrics:
            row.append(long_data[town].get(met, ""))
        # Tiny routes metrics (reordered: DS_nb, BR, N_nb, DS)
        for met in metrics:
            row.append(tiny_data[town].get(met, ""))
        data_rows.append(row)
    
    fname = "study_table_rq2_generalization_carla_detail.csv"
    write_csv_multiheader(out_dir / fname, header_rows=[h1, h2], data_rows=data_rows)
    print(f"[OK] Built CARLA generalization detail table: {len(data_rows)} rows (towns)")


# ==============================================================================
# PART 2: LaTeX Generation
# ==============================================================================
def escape_latex(x: Any) -> str:
    s = "" if x is None else str(x)
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("$", "\\$")
         .replace("#", "\\#")
         .replace("_", "\\_")
         .replace("{", "\\{")
         .replace("}", "\\}")
         .replace("~", "\\textasciitilde{}")
         .replace("^", "\\textasciicircum{}")
    )

def read_csv_for_latex(path: Path) -> List[List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.reader(f)]

def is_number(s: Any) -> bool:
    try:
        if s is None:
            return False
        t = str(s).strip()
        if t == "" or t.lower() in {"nan", "none", "--"}:
            return False
        float(t)
        return True
    except Exception:
        return False

def fmt_num(x: Any, digits: int) -> str:
    t = "" if x is None else str(x).strip()
    if t == "" or t.lower() in {"nan", "none", "--"}:
        return "--"
    try:
        v = float(t)
    except Exception:
        return escape_latex(t)
    if not math.isfinite(v):
        return "--"
    if digits == 0:
        return f"{v:.0f}"
    return f"{v:.{digits}f}"

def fmt_percent(x: Any, digits: int = 1) -> str:
    t = "" if x is None else str(x).strip()
    if t == "" or t.lower() in {"nan", "none", "--"}:
        return "--"
    try:
        v = float(t)
    except Exception:
        return escape_latex(t)
    if not math.isfinite(v):
        return "--"
    return f"{(v * 100.0):.{digits}f}"

def get_neon_bg_color_hex(val: float, p05: float, p95: float, invert: bool = False) -> str:
    if math.isnan(val) or math.isinf(val):
        return "FFFFFF"
    norm_val = max(p05, min(p95, val))
    if p05 == p95:
        norm = 0.5
    else:
        norm = 1.0 - (norm_val - p05) / (p95 - p05) if invert else (norm_val - p05) / (p95 - p05)
    if norm < 0.5:
        ratio = norm * 2.0
        r, g, b = 255, int(102 + (153 * ratio)), 102
    else:
        ratio = (norm - 0.5) * 2.0
        r, g, b = int(255 - (153 * ratio)), 255, 102
    return f"{r:02X}{g:02X}{b:02X}"

def wrap_table_content(tabular_latex: str) -> str:
    return "\\sbox0{%\n" + tabular_latex + "\n}\n\\ifdim\\wd0>\\linewidth\n  \\resizebox{\\linewidth}{!}{\\usebox0}\n\\else\n  \\usebox0\n\\fi"

def write_normal_table(rows: List[List[str]], caption: str, label: str, digits: int) -> str:
    header = rows[0]
    data = rows[1:]
    ncol = len(header)
    colspec = ("l" * min(3, ncol)) + ("r" * max(0, ncol - 3))
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    inner: List[str] = []
    inner.append("\\small")
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    inner.append(" & ".join(escape_latex(h) for h in header) + " \\\\")
    inner.append("\\midrule")
    for r in data:
        cells: List[str] = []
        for j in range(ncol):
            c = r[j] if j < len(r) else ""
            if j >= 3 and is_number(c):
                cells.append(fmt_num(c, digits))
            else:
                cells.append(escape_latex(c))
        inner.append(" & ".join(cells) + " \\\\")
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    out.append(wrap_table_content("\n".join(inner)))
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)

def write_carla_generalization_subtable(sub_data: List[List[str]], header: List[str], metric_map: Dict[str, int], invert_map: Dict[str, bool], digits: int) -> str:
    """
    Write CARLA generalization subtable with proper double headers for Long Routes vs Tiny Routes.
    """
    if not sub_data:
        return ""
    
    # Calculate statistics for coloring
    stats: Dict[str, Tuple[float, float]] = {}
    for m_name, m_idx in metric_map.items():
        vals = [float(r[m_idx]) for r in sub_data if m_idx < len(r) and is_number(r[m_idx])]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            stats[m_name] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats[m_name] = (0.0, 0.0)
    
    # Sort by DS_nb (long routes) descending, then BR ascending
    def sort_key(r: List[str]) -> Tuple[float, float, float]:
        vals: Dict[str, float] = {}
        for k, idx in metric_map.items():
            vals[k] = float(r[idx]) if (idx < len(r) and is_number(r[idx])) else float("-inf")
        br_val = vals.get("BR", float("inf"))
        ds_val = vals.get("DS", float("-inf"))
        return (vals.get("DS_nb", float("-inf")), -br_val, ds_val)
    
    sub_data.sort(key=sort_key, reverse=True)
    
    inner: List[str] = []
    inner.append("\\small")
    
    # Column specification: Model | Long Routes (4 cols) | Tiny Routes (4 cols)
    colspec = "l|cccc|cccc"
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    
    # Title row
    inner.append("\\multicolumn{9}{c}{\\textbf{CARLA}} \\\\")
    inner.append("\\midrule")
    
    # Route type header row
    inner.append("Model & \\multicolumn{4}{c|}{Long Routes} & \\multicolumn{4}{c}{Tiny Routes} \\\\")
    
    # Metric header row (reordered: DS_nb, BR, N_nb, DS)
    metric_labels = ["$DS_{nb}$", "BR [\\%]", "$N_{nb}$", "DS"]
    inner.append(" & " + " & ".join(metric_labels) + " & " + " & ".join(metric_labels) + " \\\\")
    inner.append("\\midrule")
    
    # Data rows
    for r in sub_data:
        cells: List[str] = []
        
        # Model name
        try:
            model_idx = header.index("model")
            model_name = r[model_idx] if model_idx < len(r) else ""
            cells.append(escape_latex(model_name))
        except ValueError:
            cells.append("--")
        
        # Long routes metrics (reordered: DS_nb, BR, N_nb, DS)
        long_metrics = ["DS_nb", "BR", "N_nb", "DS"]
        for col_name in long_metrics:
            try:
                idx = header.index(col_name)
                val = r[idx] if idx < len(r) else ""
                if col_name in metric_map and is_number(val):
                    v_float = float(val)
                    p05, p95 = stats.get(col_name, (0.0, 0.0))
                    inv = invert_map.get(col_name, False)
                    hex_code = get_neon_bg_color_hex(v_float, p05, p95, invert=inv)
                    if col_name == "BR":
                        disp = fmt_percent(val, 1)
                    elif col_name == "N_nb":
                        disp = fmt_num(val, 0)
                    else:
                        disp = fmt_num(val, digits)
                    cells.append(f"\\cellcolor[HTML]{{{hex_code}}}{disp}")
                else:
                    cells.append("--")
            except ValueError:
                cells.append("--")
        
        # Tiny routes metrics (reordered: DS_nb_tiny, BR_tiny, N_nb_tiny, DS_tiny)
        tiny_metrics = ["DS_nb_tiny", "BR_tiny", "N_nb_tiny", "DS_tiny"]
        for col_name in tiny_metrics:
            try:
                idx = header.index(col_name)
                val = r[idx] if idx < len(r) else ""
                if col_name in metric_map and is_number(val):
                    v_float = float(val)
                    p05, p95 = stats.get(col_name, (0.0, 0.0))
                    inv = invert_map.get(col_name, False)
                    hex_code = get_neon_bg_color_hex(v_float, p05, p95, invert=inv)
                    if col_name == "BR_tiny":
                        disp = fmt_percent(val, 1)
                    elif col_name == "N_nb_tiny":
                        disp = fmt_num(val, 0)
                    else:
                        disp = fmt_num(val, digits)
                    cells.append(f"\\cellcolor[HTML]{{{hex_code}}}{disp}")
                else:
                    cells.append("--")
            except ValueError:
                cells.append("--")
        
        inner.append(" & ".join(cells) + " \\\\")
    
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    
    result = wrap_table_content("\n".join(inner)) + "\n\\vspace{0.5em}"
    return result


def write_split_latex(rows: List[List[str]], caption: str, label: str, digits: int, is_robustness: bool = True) -> str:
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    try:
        idx_map = header.index("Map") if "Map" in header else header.index("map")
    except ValueError:
        return "% Error: Table missing Map/map column"
    genroads_data = [r for r in data if "genroads" in str(r[idx_map]).lower()]
    jungle_data = [r for r in data if "jungle" in str(r[idx_map]).lower()]
    carla_data = [r for r in data if "carla" in str(r[idx_map]).lower()]
    metric_map: Dict[str, int] = {}
    for i, h in enumerate(header):
        if h in ["PR", "E_dt", "E_e", "DS_nb", "BR", "N_nb", "DS", "DS_nb_tiny", "BR_tiny", "N_nb_tiny", "DS_tiny"]:
            metric_map[h] = i
    invert_map = {"PR": False, "DS_nb": False, "BR": True, "N_nb": False, "DS": False, "E_dt": True, "E_e": True, "DS_nb_tiny": False, "BR_tiny": True, "N_nb_tiny": False, "DS_tiny": False}
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    def write_subtable(sub_data: List[List[str]], title: str, cols_to_show: List[Tuple[str, str]]) -> None:
        if not sub_data:
            return
        def showing_col(cname: str) -> bool:
            return any(c[0] == cname for c in cols_to_show)
        def sort_key(r: List[str]) -> Tuple[float, float, float]:
            vals: Dict[str, float] = {}
            for k, idx in metric_map.items():
                vals[k] = float(r[idx]) if (idx < len(r) and is_number(r[idx])) else float("-inf")
            if showing_col("DS"):
                # CARLA: sort by DS_nb first, then BR (lower is better), then DS
                br_val = vals.get("BR", float("inf"))
                return (vals.get("DS_nb", float("-inf")), -br_val, vals.get("DS", float("-inf")))
            v_pr = vals.get("PR", float("-inf"))
            raw_dt = vals.get("E_dt", float("inf"))
            v_dt = -raw_dt if raw_dt != float("inf") else float("-inf")
            raw_e = vals.get("E_e", float("inf"))
            v_e = -raw_e if raw_e != float("inf") else float("-inf")
            return (v_pr, v_dt, v_e)
        sub_data.sort(key=sort_key, reverse=True)
        stats: Dict[str, Tuple[float, float]] = {}
        for m_name, m_idx in metric_map.items():
            vals = [float(r[m_idx]) for r in sub_data if m_idx < len(r) and is_number(r[m_idx])]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                stats[m_name] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
            else:
                stats[m_name] = (0.0, 0.0)
        inner: List[str] = []
        inner.append("\\small")
        colspec = "l" + "c" * (len(cols_to_show) - 1)
        inner.append(f"\\begin{{tabular}}{{{colspec}}}")
        inner.append("\\toprule")
        inner.append(f"\\multicolumn{{{len(cols_to_show)}}}{{c}}{{\\textbf{{{title}}}}} \\\\")
        inner.append("\\midrule")
        inner.append(" & ".join([c[1] for c in cols_to_show]) + " \\\\")
        inner.append("\\midrule")
        for r in sub_data:
            cells: List[str] = []
            for col_name, _label in cols_to_show:
                try:
                    idx = header.index(col_name)
                except ValueError:
                    cells.append("--")
                    continue
                val = r[idx] if idx < len(r) else ""
                if col_name in metric_map:
                    if not is_number(val):
                        cells.append(escape_latex(val))
                    else:
                        v_float = float(val)
                        p05, p95 = stats.get(col_name, (0.0, 0.0))
                        inv = invert_map.get(col_name, False)
                        hex_code = get_neon_bg_color_hex(v_float, p05, p95, invert=inv)
                        if col_name in ["BR", "BR_tiny"]:
                            disp = fmt_percent(val, 1)
                        elif col_name in ["N_nb", "N_nb_tiny"]:
                            disp = fmt_num(val, 0)
                        else:
                            disp = fmt_num(val, digits)
                        cells.append(f"\\cellcolor[HTML]{{{hex_code}}}{disp}")
                elif col_name in ["Severity", "sev"]:
                    try:
                        s = int(float(val))
                    except Exception:
                        s = 0
                    s_hex = get_neon_bg_color_hex(float(s), 0.0, 4.0, invert=True)
                    cells.append(f"\\cellcolor[HTML]{{{s_hex}}}{s}")
                else:
                    cells.append(escape_latex(val))
            inner.append(" & ".join(cells) + " \\\\")
        inner.append("\\bottomrule")
        inner.append("\\end{tabular}%")
        out.append(wrap_table_content("\n".join(inner)))
        out.append("\\vspace{0.5em}")
    if is_robustness:
        udacity_cols = [("Model", "Model"), ("Severity", "Sev"), ("PR", "PR"), ("E_dt", "$\\mathbb{E}[d_t]$"), ("E_e", "$\\mathbb{E}[|e_t|]$")]
        carla_cols = [("Model", "Model"), ("Severity", "Sev"), ("DS_nb", "$DS_{nb}$"), ("BR", "BR [\\%]"), ("N_nb", "$N_{nb}$"), ("DS", "DS")]
        write_subtable(genroads_data, "GenRoads", udacity_cols)
        write_subtable(jungle_data, "Jungle", udacity_cols)
        write_subtable(carla_data, "CARLA", carla_cols)
    else:
        udacity_cols = [("model", "Model"), ("PR", "PR"), ("E_dt", "$\\mathbb{E}[d_t]$"), ("E_e", "$\\mathbb{E}[|e_t|]$")]
        # For CARLA generalization, we need special handling for the double header
        if carla_data:
            write_subtable(genroads_data, "GenRoads", udacity_cols)
            write_subtable(jungle_data, "Jungle", udacity_cols)
            carla_content = write_carla_generalization_subtable(carla_data, header, metric_map, invert_map, digits)
            out.append(carla_content)
        else:
            carla_cols = [("model", "Model"), ("DS_nb", "$DS_{nb}$"), ("BR", "BR [\\%]"), ("N_nb", "$N_{nb}$"), ("DS", "DS"), ("DS_nb_tiny", "$DS_{nb,tiny}$"), ("BR_tiny", "BR$_{tiny}$ [\\%]"), ("N_nb_tiny", "$N_{nb,tiny}$"), ("DS_tiny", "DS$_{tiny}$")]
            write_subtable(genroads_data, "GenRoads", udacity_cols)
            write_subtable(jungle_data, "Jungle", udacity_cols)
            write_subtable(carla_data, "CARLA", carla_cols)
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)

def write_carla_pivot_latex(rows: List[List[str]], caption: str, label: str, digits: int) -> str:
    data = rows[2:]
    colspec = "lc|cccc"
    # Reordered columns: DS_nb=2, BR=3, N_nb=4, DS=5
    cols_to_color = [2, 3, 4, 5]
    invert_cols = [3]  # BR is inverted (index 3)
    
    # Calculate stats for coloring
    stats: Dict[int, Tuple[float, float]] = {}
    for c_idx in cols_to_color:
        vals = [float(r[c_idx]) for r in data if c_idx < len(r) and is_number(r[c_idx])]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            stats[c_idx] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats[c_idx] = (0.0, 0.0)
    
    # Sort by DS_nb (column 2), then BR (column 3), then DS (column 5)
    def sort_key(r: List[str]) -> Tuple[float, float, float]:
        def _f(v: Any, default: float = float("-inf")) -> float:
            return float(v) if is_number(v) else default
        ds_nb = _f(r[2] if len(r) > 2 else "")
        br = _f(r[3] if len(r) > 3 else "", float("inf"))
        ds = _f(r[5] if len(r) > 5 else "")
        return (ds_nb, -br, ds)  # DS_nb desc, BR asc (negated), DS desc
    
    data.sort(key=sort_key, reverse=True)
    
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    inner: List[str] = []
    inner.append("\\tiny")
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    # Reordered header: DS_nb, BR, N_nb, DS
    inner.append("Perturbation & Sev & $DS_{nb}$ & BR [\\%] & $N_{nb}$ & DS \\\\")
    inner.append("\\midrule")
    
    for r in data:
        cells: List[str] = []
        cells.append(escape_latex(r[0] if len(r) > 0 else ""))
        sev_raw = r[1] if len(r) > 1 else "0"
        try:
            sev = float(sev_raw)
        except Exception:
            sev = 0.0
        sev_col = get_neon_bg_color_hex(sev, 0.0, 4.0, invert=True)
        cells.append(f"\\cellcolor[HTML]{{{sev_col}}}{fmt_num(sev_raw, 0)}")
        
        # Process columns in new order: DS_nb(2), BR(3), N_nb(4), DS(5)
        for i in [2, 3, 4, 5]:
            if i < len(r) and is_number(r[i]):
                val = float(r[i])
                p05, p95 = stats.get(i, (0.0, 0.0))
                inv = i in invert_cols
                hex_c = get_neon_bg_color_hex(val, p05, p95, invert=inv)
                if i == 3:  # BR
                    disp = fmt_percent(r[i], 1)
                elif i == 4:  # N_nb
                    disp = fmt_num(r[i], 0)
                else:  # DS_nb, DS
                    disp = fmt_num(r[i], digits)
                cells.append(f"\\cellcolor[HTML]{{{hex_c}}}{disp}")
            else:
                cells.append("--")
        inner.append(" & ".join(cells) + " \\\\")
    
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    out.append(wrap_table_content("\n".join(inner)))
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def looks_like_single_map_pivot(rows: List[List[str]]) -> bool:
    if len(rows) < 4:
        return False
    h2 = rows[2]
    if len(h2) < 2:
        return False
    return str(h2[0]).strip().lower() == "perturbation"

def looks_like_genroads_gen_detail(rows: List[List[str]]) -> bool:
    """Check if this is a GenRoads generalization detail table (road only)."""
    if len(rows) < 4:
        return False
    h3 = rows[2]
    if len(h3) < 1:
        return False
    # Now we just have 'road' as the first column
    return str(h3[0]).strip().lower() == "road"


def looks_like_genroads_gen_scenarios(rows: List[List[str]]) -> bool:
    """Check if this is a GenRoads generalization scenarios table (scenario only)."""
    if len(rows) < 4:
        print(f"DEBUG: genroads_gen_scenarios - not enough rows: {len(rows)}")
        return False
    h3 = rows[2]
    if len(h3) < 1:
        print(f"DEBUG: genroads_gen_scenarios - empty row 3")
        return False
    # Now we just have 'scenario' as the first column
    first_col = str(h3[0]).strip().lower()
    result = first_col == "scenario"
    print(f"DEBUG: genroads_gen_scenarios - first_col='{first_col}', result={result}")
    return result

def looks_like_jungle_gen_detail(rows: List[List[str]]) -> bool:
    """Check if this is a Jungle generalization detail table (segment)."""
    if len(rows) < 4:
        return False
    h3 = rows[2]
    if len(h3) < 1:
        return False
    return str(h3[0]).strip().lower() == "segment"


def looks_like_jungle_rob_detail(rows: List[List[str]]) -> bool:
    """Check if this is a Jungle robustness detail table (segment)."""
    if len(rows) < 4:
        print(f"DEBUG: jungle_rob_detail - not enough rows: {len(rows)}")
        return False
    h3 = rows[2]
    if len(h3) < 1:
        print(f"DEBUG: jungle_rob_detail - empty row 3")
        return False
    # Check if it's a segment table and if the filename suggests robustness
    first_col = str(h3[0]).strip().lower()
    result = first_col == "segment"
    print(f"DEBUG: jungle_rob_detail - first_col='{first_col}', result={result}")
    return result

def looks_like_carla_gen_detail(rows: List[List[str]]) -> bool:
    """Check if this is a CARLA generalization detail table (Town, Long/Tiny routes)."""
    if len(rows) < 3:
        return False
    h1 = rows[0]
    h2 = rows[1]
    # Check for Long Routes and Tiny Routes in header
    h1_joined = " ".join(str(x).lower() for x in h1)
    return "long routes" in h1_joined and "tiny routes" in h1_joined

def write_single_map_pivot_latex(rows: List[List[str]], label: str, digits: int) -> str:
    map_name_raw = rows[0][2] if rows and rows[0] and len(rows[0]) > 2 else ""
    map_display = "GenRoads" if "genroads" in map_name_raw.lower() else ("Jungle" if "jungle" in map_name_raw.lower() else map_name_raw)
    caption = f"Robustness results per perturbation and model ({map_display}) (sorted descending by performance)"
    pr_zero_decimals = "genroads" in map_name_raw.lower()
    data = rows[3:]
    models = ["dave2", "dave2_gru", "vit"]
    metrics = ["PR", "E_dt", "E_e"]
    metric_label = {"PR": r"$\mathrm{PR}$", "E_dt": r"$\mathbb{E}[d_t]$", "E_e": r"$\mathbb{E}[|e_t|]$"}
    start_col = 2
    stats: Dict[int, Tuple[float, float]] = {}
    for m_idx in range(len(metrics)):
        vals: List[float] = []
        col_indices = [start_col + m_idx + (len(metrics) * mdl_i) for mdl_i in range(len(models))]
        for r in data:
            for c in col_indices:
                if c < len(r) and is_number(r[c]):
                    vals.append(float(r[c]))
        if vals:
            arr = np.array(vals, dtype=np.float64)
            stats[m_idx] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats[m_idx] = (0.0, 0.0)
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    inner: List[str] = []
    inner.append("\\tiny")
    colspec = "lc|" + ("c" * len(metrics) + "|") * (len(models) - 1) + ("c" * len(metrics))
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    h_models = ["", ""]
    for i, mdl in enumerate(models):
        trailing = "|" if i != (len(models) - 1) else ""
        h_models.append(rf"\multicolumn{{{len(metrics)}}}{{c{trailing}}}{{{escape_latex(mdl)}}}")
    inner.append(" & ".join(h_models) + " \\\\")
    h_metrics = [escape_latex("perturbation"), escape_latex("sev")]
    for _ in models:
        for met in metrics:
            h_metrics.append(metric_label.get(met, escape_latex(met)))
    inner.append(" & ".join(h_metrics) + " \\\\")
    inner.append("\\midrule")
    for r in data:
        pert = r[0] if len(r) > 0 else ""
        sev_raw = r[1] if len(r) > 1 else "0"
        try:
            sev = float(sev_raw)
        except Exception:
            sev = 0.0
        sev_col = get_neon_bg_color_hex(sev, 0.0, 4.0, invert=True)
        row_cells: List[str] = [escape_latex(pert), f"\\cellcolor[HTML]{{{sev_col}}}{fmt_num(sev_raw, 0)}"]
        for k_mdl in range(len(models)):
            for k_met in range(len(metrics)):
                col_idx = start_col + (k_mdl * len(metrics)) + k_met
                val_str = r[col_idx] if col_idx < len(r) else ""
                if is_number(val_str):
                    val = float(val_str)
                    p05, p95 = stats.get(k_met, (0.0, 0.0))
                    inv = metrics[k_met] != "PR"
                    hex_c = get_neon_bg_color_hex(val, p05, p95, invert=inv)
                    if pr_zero_decimals and metrics[k_met] == "PR":
                        disp = fmt_num(val_str, 0)
                    else:
                        disp = fmt_num(val_str, digits)
                    row_cells.append(f"\\cellcolor[HTML]{{{hex_c}}}{disp}")
                else:
                    row_cells.append("--")
        inner.append(" & ".join(row_cells) + " \\\\")
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    out.append(wrap_table_content("\n".join(inner)))
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def write_genroads_gen_detail_latex(rows: List[List[str]], label: str, digits: int) -> str:
    """
    Write LaTeX for GenRoads generalization detail table.
    Rows: road
    Columns: 3 models × 3 metrics (PR, E_dt, E_e)
    Already sorted by performance in CSV generation.
    """
    caption = "Generalization results per road (GenRoads) (sorted descending by performance)"
    data = rows[3:]  # Skip 3 header rows
    models = ["dave2", "dave2_gru", "vit"]
    metrics = ["PR", "E_dt", "E_e"]
    metric_label = {"PR": r"$\mathrm{PR}$", "E_dt": r"$\mathbb{E}[d_t]$", "E_e": r"$\mathbb{E}[|e_t|]$"}
    start_col = 1  # After road only
    
    # Calculate stats for coloring
    stats: Dict[int, Tuple[float, float]] = {}
    for m_idx in range(len(metrics)):
        vals: List[float] = []
        col_indices = [start_col + m_idx + (len(metrics) * mdl_i) for mdl_i in range(len(models))]
        for r in data:
            for c in col_indices:
                if c < len(r) and is_number(r[c]):
                    vals.append(float(r[c]))
        if vals:
            arr = np.array(vals, dtype=np.float64)
            stats[m_idx] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats[m_idx] = (0.0, 0.0)
    
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    inner: List[str] = []
    inner.append("\\small")
    colspec = "l|" + ("c" * len(metrics) + "|") * (len(models) - 1) + ("c" * len(metrics))
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    
    # Model header row
    h_models = [""]
    for i, mdl in enumerate(models):
        trailing = "|" if i != (len(models) - 1) else ""
        h_models.append(rf"\multicolumn{{{len(metrics)}}}{{c{trailing}}}{{{escape_latex(mdl)}}}")
    inner.append(" & ".join(h_models) + " \\\\")
    
    # Metric header row
    h_metrics = ["Road"]
    for _ in models:
        for met in metrics:
            h_metrics.append(metric_label.get(met, escape_latex(met)))
    inner.append(" & ".join(h_metrics) + " \\\\")
    inner.append("\\midrule")
    
    # Data rows
    for r in data:
        road = r[0] if len(r) > 0 else ""
        row_cells: List[str] = [escape_latex(road)]
        
        for k_mdl in range(len(models)):
            for k_met in range(len(metrics)):
                col_idx = start_col + (k_mdl * len(metrics)) + k_met
                val_str = r[col_idx] if col_idx < len(r) else ""
                if is_number(val_str):
                    val = float(val_str)
                    p05, p95 = stats.get(k_met, (0.0, 0.0))
                    inv = metrics[k_met] != "PR"
                    hex_c = get_neon_bg_color_hex(val, p05, p95, invert=inv)
                    disp = fmt_num(val_str, digits)
                    row_cells.append(f"\\cellcolor[HTML]{{{hex_c}}}{disp}")
                else:
                    row_cells.append("--")
        inner.append(" & ".join(row_cells) + " \\\\")
    
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    out.append(wrap_table_content("\n".join(inner)))
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def write_genroads_gen_scenarios_latex(rows: List[List[str]], label: str, digits: int) -> str:
    """
    Write LaTeX for GenRoads generalization scenarios table.
    Rows: scenario type
    Columns: 3 models × 3 metrics (PR, E_dt, E_e)
    Already sorted by performance in CSV generation.
    """
    caption = "Generalization results per scenario type (GenRoads) (sorted descending by performance)"
    data = rows[3:]  # Skip 3 header rows
    models = ["dave2", "dave2_gru", "vit"]
    metrics = ["PR", "E_dt", "E_e"]
    metric_label = {"PR": r"$\mathrm{PR}$", "E_dt": r"$\mathbb{E}[d_t]$", "E_e": r"$\mathbb{E}[|e_t|]$"}
    start_col = 1  # After scenario only
    
    # Calculate stats for coloring
    stats: Dict[int, Tuple[float, float]] = {}
    for m_idx in range(len(metrics)):
        vals: List[float] = []
        col_indices = [start_col + m_idx + (len(metrics) * mdl_i) for mdl_i in range(len(models))]
        for r in data:
            for c in col_indices:
                if c < len(r) and is_number(r[c]):
                    vals.append(float(r[c]))
        if vals:
            arr = np.array(vals, dtype=np.float64)
            stats[m_idx] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats[m_idx] = (0.0, 0.0)
    
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    inner: List[str] = []
    inner.append("\\small")
    colspec = "l|" + ("c" * len(metrics) + "|") * (len(models) - 1) + ("c" * len(metrics))
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    
    # Model header row
    h_models = [""]
    for i, mdl in enumerate(models):
        trailing = "|" if i != (len(models) - 1) else ""
        h_models.append(rf"\multicolumn{{{len(metrics)}}}{{c{trailing}}}{{{escape_latex(mdl)}}}")
    inner.append(" & ".join(h_models) + " \\\\")
    
    # Metric header row
    h_metrics = ["Scenario"]
    for _ in models:
        for met in metrics:
            h_metrics.append(metric_label.get(met, escape_latex(met)))
    inner.append(" & ".join(h_metrics) + " \\\\")
    inner.append("\\midrule")
    
    # Data rows
    for r in data:
        scenario = r[0] if len(r) > 0 else ""
        row_cells: List[str] = [escape_latex(scenario)]
        
        for k_mdl in range(len(models)):
            for k_met in range(len(metrics)):
                col_idx = start_col + (k_mdl * len(metrics)) + k_met
                val_str = r[col_idx] if col_idx < len(r) else ""
                if is_number(val_str):
                    val = float(val_str)
                    p05, p95 = stats.get(k_met, (0.0, 0.0))
                    inv = metrics[k_met] != "PR"
                    hex_c = get_neon_bg_color_hex(val, p05, p95, invert=inv)
                    disp = fmt_num(val_str, digits)
                    row_cells.append(f"\\cellcolor[HTML]{{{hex_c}}}{disp}")
                else:
                    row_cells.append("--")
        inner.append(" & ".join(row_cells) + " \\\\")
    
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    out.append(wrap_table_content("\n".join(inner)))
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def write_jungle_gen_detail_latex(rows: List[List[str]], label: str, digits: int) -> str:
    """
    Write LaTeX for Jungle generalization detail table.
    Rows: segment
    Columns: 3 models × 3 metrics (PR, E_dt, E_e)
    Already sorted by performance in CSV generation.
    """
    caption = "Generalization results per segment (Jungle) (sorted descending by performance)"
    data = rows[3:]  # Skip 3 header rows
    models = ["dave2", "dave2_gru", "vit"]
    metrics = ["PR", "E_dt", "E_e"]
    metric_label = {"PR": r"$\mathrm{PR}$", "E_dt": r"$\mathbb{E}[d_t]$", "E_e": r"$\mathbb{E}[|e_t|]$"}
    start_col = 1  # After segment
    
    # Calculate stats for coloring
    stats: Dict[int, Tuple[float, float]] = {}
    for m_idx in range(len(metrics)):
        vals: List[float] = []
        col_indices = [start_col + m_idx + (len(metrics) * mdl_i) for mdl_i in range(len(models))]
        for r in data:
            for c in col_indices:
                if c < len(r) and is_number(r[c]):
                    vals.append(float(r[c]))
        if vals:
            arr = np.array(vals, dtype=np.float64)
            stats[m_idx] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats[m_idx] = (0.0, 0.0)
    
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    inner: List[str] = []
    inner.append("\\small")
    colspec = "l|" + ("c" * len(metrics) + "|") * (len(models) - 1) + ("c" * len(metrics))
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    
    # Model header row
    h_models = [""]
    for i, mdl in enumerate(models):
        trailing = "|" if i != (len(models) - 1) else ""
        h_models.append(rf"\multicolumn{{{len(metrics)}}}{{c{trailing}}}{{{escape_latex(mdl)}}}")
    inner.append(" & ".join(h_models) + " \\\\")
    
    # Metric header row
    h_metrics = ["Segment"]
    for _ in models:
        for met in metrics:
            h_metrics.append(metric_label.get(met, escape_latex(met)))
    inner.append(" & ".join(h_metrics) + " \\\\")
    inner.append("\\midrule")
    
    # Data rows
    for r in data:
        segment = r[0] if len(r) > 0 else ""
        row_cells: List[str] = [escape_latex(segment)]
        
        for k_mdl in range(len(models)):
            for k_met in range(len(metrics)):
                col_idx = start_col + (k_mdl * len(metrics)) + k_met
                val_str = r[col_idx] if col_idx < len(r) else ""
                if is_number(val_str):
                    val = float(val_str)
                    p05, p95 = stats.get(k_met, (0.0, 0.0))
                    inv = metrics[k_met] != "PR"
                    hex_c = get_neon_bg_color_hex(val, p05, p95, invert=inv)
                    disp = fmt_num(val_str, digits)
                    row_cells.append(f"\\cellcolor[HTML]{{{hex_c}}}{disp}")
                else:
                    row_cells.append("--")
        inner.append(" & ".join(row_cells) + " \\\\")
    
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    out.append(wrap_table_content("\n".join(inner)))
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def write_jungle_rob_detail_latex(rows: List[List[str]], label: str, digits: int) -> str:
    """
    Write LaTeX for Jungle robustness detail table.
    Rows: segment
    Columns: 3 models × 3 metrics (PR, E_dt, E_e)
    Already sorted by performance in CSV generation.
    """
    caption = "Robustness results per segment (Jungle) (sorted descending by performance)"
    data = rows[3:]  # Skip 3 header rows
    models = ["dave2", "dave2_gru", "vit"]
    metrics = ["PR", "E_dt", "E_e"]
    metric_label = {"PR": r"$\mathrm{PR}$", "E_dt": r"$\mathbb{E}[d_t]$", "E_e": r"$\mathbb{E}[|e_t|]$"}
    start_col = 1  # After segment
    
    # Calculate stats for coloring
    stats: Dict[int, Tuple[float, float]] = {}
    for m_idx in range(len(metrics)):
        vals: List[float] = []
        col_indices = [start_col + m_idx + (len(metrics) * mdl_i) for mdl_i in range(len(models))]
        for r in data:
            for c in col_indices:
                if c < len(r) and is_number(r[c]):
                    vals.append(float(r[c]))
        if vals:
            arr = np.array(vals, dtype=np.float64)
            stats[m_idx] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats[m_idx] = (0.0, 0.0)
    
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    inner: List[str] = []
    inner.append("\\small")
    colspec = "l|" + ("c" * len(metrics) + "|") * (len(models) - 1) + ("c" * len(metrics))
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    
    # Model header row
    h_models = [""]
    for i, mdl in enumerate(models):
        trailing = "|" if i != (len(models) - 1) else ""
        h_models.append(rf"\multicolumn{{{len(metrics)}}}{{c{trailing}}}{{{escape_latex(mdl)}}}")
    inner.append(" & ".join(h_models) + " \\\\")
    
    # Metric header row
    h_metrics = ["Segment"]
    for _ in models:
        for met in metrics:
            h_metrics.append(metric_label.get(met, escape_latex(met)))
    inner.append(" & ".join(h_metrics) + " \\\\")
    inner.append("\\midrule")
    
    # Data rows
    for r in data:
        segment = r[0] if len(r) > 0 else ""
        row_cells: List[str] = [escape_latex(segment)]
        
        for k_mdl in range(len(models)):
            for k_met in range(len(metrics)):
                col_idx = start_col + (k_mdl * len(metrics)) + k_met
                val_str = r[col_idx] if col_idx < len(r) else ""
                if is_number(val_str):
                    val = float(val_str)
                    p05, p95 = stats.get(k_met, (0.0, 0.0))
                    inv = metrics[k_met] != "PR"
                    hex_c = get_neon_bg_color_hex(val, p05, p95, invert=inv)
                    disp = fmt_num(val_str, digits)
                    row_cells.append(f"\\cellcolor[HTML]{{{hex_c}}}{disp}")
                else:
                    row_cells.append("--")
        inner.append(" & ".join(row_cells) + " \\\\")
    
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    out.append(wrap_table_content("\n".join(inner)))
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def write_carla_gen_detail_latex(rows: List[List[str]], label: str, digits: int) -> str:
    """
    Write LaTeX for CARLA generalization detail table.
    Rows: Town
    Columns: Long Routes (DS_nb, BR, N_nb, DS) | Tiny Routes (DS_nb, BR, N_nb, DS)
    Already sorted by performance in CSV generation.
    """
    caption = "Generalization results per town (CARLA): Long Routes vs Tiny Routes (nominal) (sorted descending by performance)"
    data = rows[2:]  # Skip 2 header rows
    metrics = ["DS", "BR", "N_nb", "DS_nb"]
    
    # Calculate stats for coloring (separate for long and tiny) - reordered metrics
    stats_long: Dict[int, Tuple[float, float]] = {}
    stats_tiny: Dict[int, Tuple[float, float]] = {}
    
    # Use the correct indices that match the CSV structure and rendering
    long_indices = [1, 2, 3, 4]  # DS_nb, BR, N_nb, DS (matches CSV and rendering)
    tiny_indices = [5, 6, 7, 8]  # DS_nb_tiny, BR_tiny, N_nb_tiny, DS_tiny (matches CSV and rendering)
    
    for new_idx, old_idx in enumerate(long_indices):
        vals_long = [float(r[old_idx]) for r in data if old_idx < len(r) and is_number(r[old_idx])]
        if vals_long:
            arr = np.array(vals_long, dtype=np.float64)
            stats_long[new_idx] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats_long[new_idx] = (0.0, 0.0)
    
    for new_idx, old_idx in enumerate(tiny_indices):
        vals_tiny = [float(r[old_idx]) for r in data if old_idx < len(r) and is_number(r[old_idx])]
        if vals_tiny:
            arr = np.array(vals_tiny, dtype=np.float64)
            stats_tiny[new_idx] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        else:
            stats_tiny[new_idx] = (0.0, 0.0)
    
    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    inner: List[str] = []
    inner.append("\\small")
    colspec = "l|cccc|cccc"
    inner.append(f"\\begin{{tabular}}{{{colspec}}}")
    inner.append("\\toprule")
    
    # Route type header row
    inner.append(r"& \multicolumn{4}{c|}{Long Routes} & \multicolumn{4}{c}{Tiny Routes} \\")
    
    # Metric header row (reordered: DS_nb, BR, N_nb, DS)
    metric_labels = ["$DS_{nb}$", "BR [\\%]", "$N_{nb}$", "DS"]
    inner.append("Town & " + " & ".join(metric_labels) + " & " + " & ".join(metric_labels) + " \\\\")
    inner.append("\\midrule")
    
    # Data rows
    for r in data:
        town = r[0] if len(r) > 0 else ""
        row_cells: List[str] = [escape_latex(town)]
        
        # Long routes metrics (correct order: DS_nb=1, BR=2, N_nb=3, DS=4)
        long_indices = [1, 2, 3, 4]  # DS_nb, BR, N_nb, DS
        for i, col_idx in enumerate(long_indices):
            val_str = r[col_idx] if col_idx < len(r) else ""
            if is_number(val_str):
                val = float(val_str)
                p05, p95 = stats_long.get(i, (0.0, 0.0))
                inv = i == 1  # BR is inverted (index 1 in reordered list)
                hex_c = get_neon_bg_color_hex(val, p05, p95, invert=inv)
                if i == 1:  # BR
                    disp = fmt_percent(val_str, 1)
                elif i == 2:  # N_nb
                    disp = fmt_num(val_str, 0)
                else:
                    disp = fmt_num(val_str, digits)
                row_cells.append(f"\\cellcolor[HTML]{{{hex_c}}}{disp}")
            else:
                row_cells.append("--")
        
        # Tiny routes metrics (correct order: DS_nb_tiny=5, BR_tiny=6, N_nb_tiny=7, DS_tiny=8)
        tiny_indices = [5, 6, 7, 8]  # DS_nb_tiny, BR_tiny, N_nb_tiny, DS_tiny
        for i, col_idx in enumerate(tiny_indices):
            val_str = r[col_idx] if col_idx < len(r) else ""
            if is_number(val_str):
                val = float(val_str)
                p05, p95 = stats_tiny.get(i, (0.0, 0.0))
                inv = i == 1  # BR is inverted (index 1 in reordered list)
                hex_c = get_neon_bg_color_hex(val, p05, p95, invert=inv)
                if i == 1:  # BR
                    disp = fmt_percent(val_str, 1)
                elif i == 2:  # N_nb
                    disp = fmt_num(val_str, 0)
                else:
                    disp = fmt_num(val_str, digits)
                row_cells.append(f"\\cellcolor[HTML]{{{hex_c}}}{disp}")
            else:
                row_cells.append("--")
        
        inner.append(" & ".join(row_cells) + " \\\\")
    
    inner.append("\\bottomrule")
    inner.append("\\end{tabular}%")
    out.append(wrap_table_content("\n".join(inner)))
    out.append(f"\\caption{{{escape_latex(caption)}}}")
    out.append(f"\\label{{{escape_latex(label)}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def convert_csvs_to_latex(in_dir: Path, out_dir: Path, digits: int) -> None:
    inputs: List[str] = []
    files = sorted(in_dir.glob("study_table_*.csv"))
    for csv_path in files:
        rows = read_csv_for_latex(csv_path)
        if not rows:
            continue
        stem = csv_path.stem
        label = f"tab:{stem}"
        
        print(f"DEBUG: Processing {stem}")
        if len(rows) >= 3:
            print(f"DEBUG: Row 3: {rows[2][:5] if len(rows[2]) > 5 else rows[2]}")
        
        # RQ1 Robustness main table
        if "rq1_robustness" in stem and "perturbation" not in stem and "detail" not in stem:
            print(f"DEBUG: Using write_split_latex for {stem}")
            tex = write_split_latex(rows, "Robustness scores per model (sorted descending by performance)", label, digits, is_robustness=True)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # RQ2 Generalization main table
        if "rq2_generalization" in stem and "detail" not in stem and "scenarios" not in stem:
            print(f"DEBUG: Using write_split_latex for {stem}")
            tex = write_split_latex(rows, "Generalization scores (sorted descending by performance)", label, digits, is_robustness=False)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # RQ1 CARLA robustness by perturbation
        if "robustness_by_perturbation_carla" in stem:
            print(f"DEBUG: Using write_carla_pivot_latex for {stem}")
            tex = write_carla_pivot_latex(rows, "Robustness results per perturbation and model (CARLA) (sorted descending by performance)", label, digits)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # RQ1 GenRoads/Jungle robustness by perturbation
        if looks_like_single_map_pivot(rows) and "generalization" not in stem:
            print(f"DEBUG: Using write_single_map_pivot_latex for {stem}")
            tex = write_single_map_pivot_latex(rows, label, digits)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # NEW: RQ2 GenRoads generalization detail table
        if looks_like_genroads_gen_detail(rows):
            print(f"DEBUG: Using write_genroads_gen_detail_latex for {stem}")
            tex = write_genroads_gen_detail_latex(rows, label, digits)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # NEW: RQ2 GenRoads generalization scenarios table
        if looks_like_genroads_gen_scenarios(rows):
            print(f"DEBUG: Using write_genroads_gen_scenarios_latex for {stem}")
            tex = write_genroads_gen_scenarios_latex(rows, label, digits)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # NEW: RQ2 Jungle generalization detail table
        if looks_like_jungle_gen_detail(rows) and "generalization" in stem:
            print(f"DEBUG: Using write_jungle_gen_detail_latex for {stem}")
            tex = write_jungle_gen_detail_latex(rows, label, digits)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # NEW: RQ1 Jungle robustness detail table
        if looks_like_jungle_rob_detail(rows) and "robustness" in stem:
            print(f"DEBUG: Using write_jungle_rob_detail_latex for {stem}")
            tex = write_jungle_rob_detail_latex(rows, label, digits)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # NEW: RQ2 CARLA generalization detail table
        if looks_like_carla_gen_detail(rows):
            print(f"DEBUG: Using write_carla_gen_detail_latex for {stem}")
            tex = write_carla_gen_detail_latex(rows, label, digits)
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            inputs.append(f"\\input{{{stem}}}")
            continue
        
        # Default: normal table
        print(f"DEBUG: Using write_normal_table for {stem}")
        tex = write_normal_table(rows, caption="Table", label=label, digits=digits)
        (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
        inputs.append(f"\\input{{{stem}}}")
    
    (out_dir / "all_tables.tex").write_text("\n".join(inputs) + "\n", encoding="utf-8")
    print(f"[OK] wrote LaTeX files to: {out_dir}")


# ==============================================================================
# CLI
# ==============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--digits", type=int, default=3)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    
    print("Building RQ1 Robustness table...")
    build_table_rq1(results_root, out_dir)
    
    print("Building RQ2 Generalization summary table...")
    build_table_rq2(results_root, out_dir)
    
    print("Building RQ1 per-perturbation pivot tables...")
    build_pivot_perturbations_independent(results_root, out_dir)
    
    print("Building RQ2 GenRoads generalization detail table...")
    build_table_rq2_genroads_detail(results_root, out_dir)
    
    print("Building RQ2 GenRoads generalization scenarios table...")
    build_table_rq2_genroads_scenarios(results_root, out_dir)
    
    print("Building RQ2 Jungle generalization detail table...")
    build_table_rq2_jungle_detail(results_root, out_dir)
    
    print("Building RQ1 Jungle robustness detail table...")
    build_table_rq1_jungle_detail(results_root, out_dir)
    
    print("Building RQ2 CARLA generalization detail table...")
    build_table_rq2_carla_detail(results_root, out_dir)
    
    print("Converting CSVs to LaTeX...")
    convert_csvs_to_latex(out_dir, out_dir, args.digits)
    
    print("Done.")

if __name__ == "__main__":
    main()