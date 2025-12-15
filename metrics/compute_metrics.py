from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

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


# -----------------------
# CONFIG (edit these)
# -----------------------

# Example Udacity:
#   RUN_DIR = Path("/media/.../runs/jungle/robustness/dave2_gru_20251214_203816")
# Example CARLA:
#   RUN_DIR = Path("/media/.../runs/carla/robustness/tcp/20251212_185734")

SIM = "carla"  # "udacity" or "carla"

# RUN_DIR = Path("/media/maximigenbergs/Elements/maximigenbergs/runs/carla/generalization/tcp/20251201_120732")
RUN_DIR = Path("/media/maximigenbergs/Elements/maximigenbergs/runs/carla/robustness/tcp/20251212_185734")

MAP_NAME = "multi-town"          # udacity: "jungle" or "genroads"; carla: keep "multi-town"
TEST_TYPE = "robustness"     # "robustness" or "generalization"

# OUT_DIR = Path("/media/maximigenbergs/Elements/maximigenbergs/results/carla/generalization/tcp/20251201_120732")
OUT_DIR = Path("/media/maximigenbergs/Elements/maximigenbergs/results/carla/robustness/tcp/20251212_185734")

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
