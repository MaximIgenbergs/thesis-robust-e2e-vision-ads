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
)
from metrics.report_tables import write_csv

SIM = "udacity" # "udacity" or "carla"

# carla generalization tcp
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/carla/generalization/tcp/20251201_120732"))
# OUT_DIR = abs_path(Path("results/carla/generalization/tcp/20251201_120732"))

# carla robustness tcp
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/carla/robustness/tcp/20251212_185734"))
# OUT_DIR = abs_path(Path("results/carla/robustness/tcp/20251212_185734"))

# genroads generalization vit
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/vit_20251217_182632"))
# OUT_DIR = abs_path(Path("results/genroads/generalization/vit_20251217_182632"))

# genroads generalization dave2
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/dave2_20251217_225857"))
# OUT_DIR = abs_path(Path("results/genroads/generalization/dave2_20251217_225857"))

# genroads generalization dave2_gru
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/generalization/dave2_gru_20251218_031746"))
# OUT_DIR = abs_path(Path("results/genroads/generalization/dave2_gru_20251218_031746"))

# genroads robustness vit
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/robustness/vit_20251208_181344"))
# OUT_DIR = abs_path(Path("results/genroads/robustness/vit_20251208_181344"))

# genroads robustness dave2
# RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/genroads/robustness/dave2_20251214_184315"))
# OUT_DIR = abs_path(Path("results/genroads/robustness/dave2_20251214_184315"))

# jungle robustness vit
RUN_DIR = abs_path(Path("/media/maxim/Elements/maximigenbergs/runs/jungle/robustness/vit_20251209_205922"))
OUT_DIR = abs_path(Path("results/jungle/robustness/vit_20251209_205922"))

MAP_NAME = "jungle" # udacity: jungle or genroads; carla: multi-town
TEST_TYPE = "robustness" # robustness or generalization

BASELINE_NAME = "baseline"


def main() -> None:
    if SIM == "udacity":
        entry_rows: List[Dict[str, Any]] = []

        for raw in iter_udacity_entries(RUN_DIR):
            entries = normalize_udacity_entry(
                raw,
                map_name=MAP_NAME,
                test_type=TEST_TYPE,
                baseline_name=BASELINE_NAME,
            )
            for ep in entries:
                # manifest null perturbation => baseline, but enforce anyway
                if ep["perturbation"] is None or str(ep["perturbation"]).strip() == "":
                    ep["perturbation"] = BASELINE_NAME
                    ep["severity"] = 0
                entry_rows.append(compute_udacity_entry_metrics(ep))

        summary_rows = aggregate_udacity_summary(entry_rows)
        robust_rows = robustness_summaries(summary_rows)
        mce_rows = mce_over_all_corruptions(robust_rows)

        write_csv(OUT_DIR / "entry_metrics.csv", entry_rows)
        write_csv(OUT_DIR / "summary_table.csv", summary_rows)
        write_csv(OUT_DIR / "robustness_summary.csv", robust_rows)
        write_csv(OUT_DIR / "mce_over_corruptions.csv", mce_rows)

        print(f"[OK] wrote Udacity metrics to: {OUT_DIR}")
        return

    if SIM == "carla":
        route_rows = normalize_carla_run(RUN_DIR, test_type=TEST_TYPE)
        summary_rows = aggregate_carla_summary(route_rows)

        write_csv(OUT_DIR / "route_metrics.csv", route_rows)
        write_csv(OUT_DIR / "summary_table.csv", summary_rows)

        if TEST_TYPE == "robustness":
            robust_rows = robustness_summaries(summary_rows)
            mce_rows = mce_over_all_corruptions(robust_rows)
            write_csv(OUT_DIR / "robustness_summary.csv", robust_rows)
            write_csv(OUT_DIR / "mce_over_corruptions.csv", mce_rows)

        print(f"[OK] wrote CARLA metrics to: {OUT_DIR}")
        return


    raise ValueError(f"Unknown SIM={SIM!r}. Use 'udacity' or 'carla'.")


if __name__ == "__main__":
    main()
