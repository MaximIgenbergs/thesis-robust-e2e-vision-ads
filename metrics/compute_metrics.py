from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from scripts import abs_path
from metrics.io_udacity import iter_udacity_entries, normalize_udacity_entry
from metrics.io_carla import normalize_carla_run
from metrics.compute_episode import compute_udacity_entry_metrics
from metrics.aggregate import (
    aggregate_udacity_summary,
    aggregate_carla_summary,
    robustness_summaries,
    mce_over_all_corruptions,
    robustness_by_severity,
    rq1_robustness_wide,
    rq2_generalization_row,
    robustness_conditions_table,
)
from metrics.report_tables import write_csv

BASELINE_NAME = "baseline"

# jungle robustness vit
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/robustness/vit_20251218_172745"))
# OUT_DIR = abs_path(Path("results/jungle/robustness/vit_20251218_172745"))

# jungle robustness dave2
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/robustness/dave2_20251219_165635"))
# OUT_DIR = abs_path(Path("results/jungle/robustness/dave2_20251219_165635"))

# jungle robustness dave2_gru
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/robustness/dave2_gru_20251220_113323"))
# OUT_DIR = abs_path(Path("results/jungle/robustness/dave2_gru_20251220_113323"))

# jungle generalization vit
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/generalization/vit_20251219_030941"))
# OUT_DIR = abs_path(Path("results/jungle/generalization/vit_20251219_030941"))

# jungle generalization dave2
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/generalization/dave2_20251220_225512"))
# OUT_DIR = abs_path(Path("results/jungle/generalization/dave2_20251220_225512"))

# jungle generalization dave2_gru
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/generalization/dave2_gru_20251221_000824"))
# OUT_DIR = abs_path(Path("results/jungle/generalization/dave2_gru_20251221_000824"))

# genroads robustness vit
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/robustness/vit_20251208_181344"))
# OUT_DIR = abs_path(Path("results/genroads/robustness/vit_20251208_181344"))

# genroads robustness dave2
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/robustness/dave2_20251214_184315"))
# OUT_DIR = abs_path(Path("results/genroads/robustness/dave2_20251214_184315"))

# genroads robustness dave2_gru
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/robustness/dave2_gru_20251221_193038"))
# OUT_DIR = abs_path(Path("results/genroads/robustness/dave2_gru_20251221_193038"))

# genroads generalization vit
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/vit_20251217_182632"))
# OUT_DIR = abs_path(Path("results/genroads/generalization/vit_20251217_182632"))

# genroads generalization dave2
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/dave2_20251217_225857"))
# OUT_DIR = abs_path(Path("results/genroads/generalization/dave2_20251217_225857"))

# genroads generalization dave2_gru
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/dave2_gru_20251218_031746"))
# OUT_DIR = abs_path(Path("results/genroads/generalization/dave2_gru_20251218_031746"))

# carla robustness tcp
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/carla/robustness/tcp/20251212_185734"))
# OUT_DIR = abs_path(Path("results/carla/robustness/tcp/20251212_185734"))

# carla generalization tcp
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/carla/generalization/tcp/20251201_120732"))
# OUT_DIR = abs_path(Path("results/carla/generalization/tcp/20251201_120732"))

# -------------------------
# Define jobs + select which to run
# -------------------------

JOBS: Dict[str, Dict[str, Any]] = {
    "jungle_robustness_vit": {
        "sim": "udacity",
        "map_name": "jungle",
        "test_type": "robustness",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/robustness/vit_20251218_172745")),
        "out_dir": abs_path(Path("results/jungle/robustness/vit_20251218_172745")),
    },
    "jungle_robustness_dave2": {
        "sim": "udacity",
        "map_name": "jungle",
        "test_type": "robustness",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/robustness/dave2_20251219_165635")),
        "out_dir": abs_path(Path("results/jungle/robustness/dave2_20251219_165635")),
    },
    "jungle_robustness_dave2_gru": {
        "sim": "udacity",
        "map_name": "jungle",
        "test_type": "robustness",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/robustness/dave2_gru_20251220_113323")),
        "out_dir": abs_path(Path("results/jungle/robustness/dave2_gru_20251220_113323")),
    },
    "jungle_generalization_vit": {
        "sim": "udacity",
        "map_name": "jungle",
        "test_type": "generalization",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/generalization/vit_20251219_030941")),
        "out_dir": abs_path(Path("results/jungle/generalization/vit_20251219_030941")),
    },
    "jungle_generalization_dave2": {
        "sim": "udacity",
        "map_name": "jungle",
        "test_type": "generalization",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/generalization/dave2_20251220_225512")),
        "out_dir": abs_path(Path("results/jungle/generalization/dave2_20251220_225512")),
    },
    "jungle_generalization_dave2_gru": {
        "sim": "udacity",
        "map_name": "jungle",
        "test_type": "generalization",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/generalization/dave2_gru_20251221_000824")),
        "out_dir": abs_path(Path("results/jungle/generalization/dave2_gru_20251221_000824")),
    },
    "genroads_robustness_vit": {
        "sim": "udacity",
        "map_name": "genroads",
        "test_type": "robustness",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/robustness/vit_20251208_181344")),
        "out_dir": abs_path(Path("results/genroads/robustness/vit_20251208_181344")),
    },
    "genroads_robustness_dave2": {
        "sim": "udacity",
        "map_name": "genroads",
        "test_type": "robustness",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/robustness/dave2_20251214_184315")),
        "out_dir": abs_path(Path("results/genroads/robustness/dave2_20251214_184315")),
    },
    "genroads_robustness_dave2_gru": {
        "sim": "udacity",
        "map_name": "genroads",
        "test_type": "robustness",
        "paired_severities": True,  # <-- special troubleshooting run
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/robustness/dave2_gru_20251221_193038")),
        "out_dir": abs_path(Path("results/genroads/robustness/dave2_gru_20251221_193038")),
    },
    "genroads_generalization_vit": {
        "sim": "udacity",
        "map_name": "genroads",
        "test_type": "generalization",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/vit_20251217_182632")),
        "out_dir": abs_path(Path("results/genroads/generalization/vit_20251217_182632")),
    },
    "genroads_generalization_dave2": {
        "sim": "udacity",
        "map_name": "genroads",
        "test_type": "generalization",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/dave2_20251217_225857")),
        "out_dir": abs_path(Path("results/genroads/generalization/dave2_20251217_225857")),
    },
    "genroads_generalization_dave2_gru": {
        "sim": "udacity",
        "map_name": "genroads",
        "test_type": "generalization",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/dave2_gru_20251218_031746")),
        "out_dir": abs_path(Path("results/genroads/generalization/dave2_gru_20251218_031746")),
    },
    "carla_robustness_tcp": {
        "sim": "carla",
        "map_name": "multi-town",
        "test_type": "robustness",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/carla/robustness/tcp/20251212_185734")),
        "out_dir": abs_path(Path("results/carla/robustness/tcp/20251212_185734")),
    },
    "carla_generalization_tcp": {
        "sim": "carla",
        "map_name": "multi-town",
        "test_type": "generalization",
        "paired_severities": False,
        "run_dir": abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/carla/generalization/tcp/20251201_120732")),
        "out_dir": abs_path(Path("results/carla/generalization/tcp/20251201_120732")),
    },
}

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
RUN_ALL = True  # if True, ignores RUN_SET and runs all jobs in JOBS


def run_job(cfg: Dict[str, Any]) -> None:
    sim = str(cfg["sim"])
    run_dir = Path(cfg["run_dir"])
    out_dir = Path(cfg["out_dir"])
    map_name = str(cfg["map_name"])
    test_type = str(cfg["test_type"])
    paired_severities = bool(cfg.get("paired_severities", False))

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
