from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from scripts import abs_path
from metrics.constants import BASELINE_NAME_DEFAULT as BASELINE_NAME
from metrics.io_udacity import iter_udacity_entries, normalize_udacity_entry
from metrics.io_carla import normalize_carla_run
from metrics.compute_episode import compute_udacity_entry_metrics
from metrics.aggregate import aggregate_udacity_summary, aggregate_carla_summary, robustness_summaries, mce_over_all_corruptions, robustness_by_severity, rq1_robustness_wide, rq2_generalization_row, robustness_conditions_table
from metrics.report_tables import write_csv

# Configuration: Set these environment variables or modify paths below
RUNS_ROOT = Path(os.getenv("RUNS_ROOT", "/media/maxim/Elements/maximigenbergs/runs"))
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results"))

RUN_ALL = True  # if True, ignores RUN_SET and runs all jobs in JOBS

# Choose what to run:
RUN_SET: List[str] = [
    "jungle_robustness_vit",
    "jungle_robustness_dave2",
    "jungle_robustness_dave2_gru",
    "jungle_generalization_vit",
    "jungle_generalization_dave2",
    "jungle_generalization_dave2_gru",
    "genroads_robustness_vit",
    "genroads_robustness_dave2",
    "genroads_robustness_dave2_gru",
    "genroads_generalization_vit",
    "genroads_generalization_dave2",
    "genroads_generalization_dave2_gru",
    "carla_robustness_tcp",
    "carla_generalization_tcp",
]

JOBS: Dict[str, Dict[str, Any]] = {
    # Jungle experiments
    "jungle_robustness_vit": _make_job_config("udacity", "jungle", "robustness", "vit", "20251218_172745"),
    "jungle_robustness_dave2": _make_job_config("udacity", "jungle", "robustness", "dave2", "20251219_165635"),
    "jungle_robustness_dave2_gru": _make_job_config("udacity", "jungle", "robustness", "dave2_gru", "20251220_113323"),
    "jungle_generalization_vit": _make_job_config("udacity", "jungle", "generalization", "vit", "20251219_030941"),
    "jungle_generalization_dave2": _make_job_config("udacity", "jungle", "generalization", "dave2", "20251220_225512"),
    "jungle_generalization_dave2_gru": _make_job_config("udacity", "jungle", "generalization", "dave2_gru", "20251221_000824"),
    
    # GenRoads experiments
    "genroads_robustness_vit": _make_job_config("udacity", "genroads", "robustness", "vit", "20251208_181344"),
    "genroads_robustness_dave2": _make_job_config("udacity", "genroads", "robustness", "dave2", "20251214_184315"),
    "genroads_robustness_dave2_gru": _make_job_config("udacity", "genroads", "robustness", "dave2_gru", "20251221_193038", paired_severities=True),  # Special case
    "genroads_generalization_vit": _make_job_config("udacity", "genroads", "generalization", "vit", "20251217_182632"),
    "genroads_generalization_dave2": _make_job_config("udacity", "genroads", "generalization", "dave2", "20251217_225857"),
    "genroads_generalization_dave2_gru": _make_job_config("udacity", "genroads", "generalization", "dave2_gru", "20251218_031746"),
    
    # CARLA experiments
    "carla_robustness_tcp": {
        "sim": "carla",
        "map_name": "multi-town",
        "test_type": "robustness",
        "paired_severities": False,
        "run_dir": abs_path(RUNS_ROOT / "carla" / "robustness" / "tcp" / "20251212_185734"),
        "out_dir": abs_path(RESULTS_ROOT / "carla" / "robustness" / "tcp" / "20251212_185734"),
    },
    "carla_generalization_tcp": {
        "sim": "carla",
        "map_name": "multi-town",
        "test_type": "generalization",
        "paired_severities": False,
        "run_dir": abs_path(RUNS_ROOT / "carla" / "generalization" / "tcp" / "20251201_120732"),
        "out_dir": abs_path(RESULTS_ROOT / "carla" / "generalization" / "tcp" / "20251201_120732"),
    },
}


def _make_job_config(sim: str, map_name: str, test_type: str, model: str, timestamp: str, 
                     paired_severities: bool = False) -> Dict[str, Any]:
    """
    Create a job configuration with consistent path structure.
    
    Args:
        sim: Simulation type ("udacity" or "carla")
        map_name: Map name ("jungle", "genroads", "multi-town")
        test_type: Test type ("robustness" or "generalization")
        model: Model name ("vit", "dave2", "dave2_gru", "tcp")
        timestamp: Timestamp string from run directory
        paired_severities: Whether to use paired severity handling (special case)
    
    Returns:
        Dictionary with job configuration including paths
    """
    run_dir = abs_path(RUNS_ROOT / map_name / test_type / f"{model}_{timestamp}")
    out_dir = abs_path(RESULTS_ROOT / map_name / test_type / f"{model}_{timestamp}")
    
    return {
        "sim": sim,
        "map_name": map_name,
        "test_type": test_type,
        "paired_severities": paired_severities,
        "run_dir": run_dir,
        "out_dir": out_dir,
    }


def run_job(cfg: Dict[str, Any]) -> None:
    """Run a metrics computation job"""
    # Input validation
    required_fields = ["sim", "run_dir", "out_dir", "map_name", "test_type"]
    for field in required_fields:
        if field not in cfg:
            raise ValueError(f"Missing required field: {field}")
    
    sim = str(cfg["sim"])
    if sim not in ("udacity", "carla"):
        raise ValueError(f"Invalid sim type: {sim}. Must be 'udacity' or 'carla'")
    
    test_type = str(cfg["test_type"])
    if test_type not in ("robustness", "generalization"):
        raise ValueError(f"Invalid test_type: {test_type}. Must be 'robustness' or 'generalization'")
    
    run_dir = Path(cfg["run_dir"])
    if not run_dir.exists():
        raise ValueError(f"run_dir does not exist: {run_dir}")
    
    out_dir = Path(cfg["out_dir"])
    map_name = str(cfg["map_name"])
    paired_severities = bool(cfg.get("paired_severities", False))
    
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    if sim == "udacity":
        entry_rows: List[Dict[str, Any]] = []

        for raw in iter_udacity_entries(run_dir):
            entries = normalize_udacity_entry(
                raw,
                map_name=map_name,
                test_type=test_type,
                baseline_name=BASELINE_NAME,
                paired_severities=paired_severities,
            )
            for ep in entries:
                if ep["perturbation"] is None or str(ep["perturbation"]).strip() == "":
                    ep["perturbation"] = BASELINE_NAME
                    ep["severity"] = 0
                entry_rows.append(compute_udacity_entry_metrics(ep))

        summary_rows = aggregate_udacity_summary(entry_rows)

        write_csv(out_dir / "entry_metrics.csv", entry_rows)
        write_csv(out_dir / "summary_table.csv", summary_rows)

        if test_type == "robustness":
            sev_rows = robustness_by_severity(summary_rows)
            rq1_rows = rq1_robustness_wide(sev_rows, severities=(0, 2, 4))
            cond_rows = robustness_conditions_table(summary_rows, severities=(2, 4))

            write_csv(out_dir / "robustness_by_severity.csv", sev_rows)
            write_csv(out_dir / "rq1_robustness_row.csv", rq1_rows)
            write_csv(out_dir / "robustness_by_perturbation.csv", cond_rows)

            robust_rows = robustness_summaries(summary_rows)
            mce_rows = mce_over_all_corruptions(robust_rows)
            write_csv(out_dir / "robustness_summary.csv", robust_rows)
            write_csv(out_dir / "mce_over_corruptions.csv", mce_rows)

            print(f"[OK] udacity robustness -> {out_dir}")
            return

        if test_type == "generalization":
            rq2_rows = rq2_generalization_row(summary_rows)
            write_csv(out_dir / "rq2_generalization_row.csv", rq2_rows)
            print(f"[OK] udacity generalization -> {out_dir}")
            return

        raise ValueError(f"Unknown test_type={test_type!r}. Use 'robustness' or 'generalization'.")

    if sim == "carla":
        route_rows = normalize_carla_run(run_dir, test_type=test_type)
        summary_rows = aggregate_carla_summary(route_rows, group_generalization_overall=True)

        write_csv(out_dir / "route_metrics.csv", route_rows)
        write_csv(out_dir / "summary_table.csv", summary_rows)

        if test_type == "robustness":
            sev_rows = robustness_by_severity(summary_rows)
            rq1_rows = rq1_robustness_wide(sev_rows, severities=(0, 2, 4))
            write_csv(out_dir / "robustness_by_severity.csv", sev_rows)
            write_csv(out_dir / "rq1_robustness_row.csv", rq1_rows)

            robust_rows = robustness_summaries(summary_rows)
            mce_rows = mce_over_all_corruptions(robust_rows)
            write_csv(out_dir / "robustness_summary.csv", robust_rows)
            write_csv(out_dir / "mce_over_corruptions.csv", mce_rows)

            print(f"[OK] carla robustness -> {out_dir}")
            return

        if test_type == "generalization":
            rq2_rows = rq2_generalization_row(summary_rows)
            write_csv(out_dir / "rq2_generalization_row.csv", rq2_rows)
            print(f"[OK] carla generalization -> {out_dir}")
            return

        raise ValueError(f"Unknown test_type={test_type!r}. Use 'robustness' or 'generalization'.")

    raise ValueError(f"Unknown sim={sim!r}. Use 'udacity' or 'carla'.")


def main() -> None:
    keys = list(JOBS.keys()) if RUN_ALL else RUN_SET
    if not keys:
        raise ValueError("No jobs selected. Set RUN_ALL=True or add names to RUN_SET.")

    for name in keys:
        if name not in JOBS:
            raise ValueError(f"Unknown job name: {name}")
        print(f"[JOB] {name}")
        run_job(JOBS[name])


if __name__ == "__main__":
    main()
