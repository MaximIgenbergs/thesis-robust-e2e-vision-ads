from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from metrics.io_udacity import iter_udacity_episodes, normalize_udacity_episode
from metrics.io_carla import iter_carla_route_records, normalize_carla_route_record
from metrics.compute_episode import compute_udacity_episode_metrics
from metrics.metrics import CarlaEpisodeSummary, carla_ds_active
from metrics.aggregate import aggregate_udacity_table, robustness_summaries, mce_over_all_corruptions
from metrics.report_tables import write_csv

SIM = "udacity" # "udacity" or "carla"
RUN_DIR = Path("runs/jungle/robustness/vit_20251203-175344")
MAP_NAME = "jungle" # "jungle" or "genroads" for udacity
TEST_TYPE = "robustness" # "robustness" or "generalization"

OUT_DIR = Path("results/jungle/robustness/vit_20251203-175344")

BASELINE_PERTURBATION_NAME = "baseline"


def main() -> None:
    if SIM == "udacity":
        episode_rows: List[Dict[str, Any]] = []

        for raw_ep in iter_udacity_episodes(RUN_DIR):
            ep = normalize_udacity_episode(
                raw_ep,
                map_name=MAP_NAME,
                test_type=TEST_TYPE,
                baseline_name=BASELINE_PERTURBATION_NAME,
            )

            # enforce baseline defaults if missing
            if not ep["perturbation"]:
                ep["perturbation"] = BASELINE_PERTURBATION_NAME
                ep["severity"] = 0
            if str(ep["perturbation"]).lower() == BASELINE_PERTURBATION_NAME and int(ep["severity"]) < 0:
                ep["severity"] = 0

            episode_rows.append(compute_udacity_episode_metrics(ep))

        summary_rows = aggregate_udacity_table(episode_rows)

        # Robustness summaries (AUC/CE) only make sense if your run contains multiple severities/perturbations.
        robust_rows = robustness_summaries(summary_rows)
        mce_rows = mce_over_all_corruptions(robust_rows)

        write_csv(OUT_DIR / "episode_metrics.csv", episode_rows)
        write_csv(OUT_DIR / "summary_table.csv", summary_rows)
        write_csv(OUT_DIR / "robustness_summary.csv", robust_rows)
        write_csv(OUT_DIR / "mce_over_corruptions.csv", mce_rows)

        print(f"[OK] wrote: {OUT_DIR.resolve()}")
        return

    if SIM == "carla":
        # One CARLA run dir => route-level + DS_active/BR summary
        route_rows: List[Dict[str, Any]] = []
        episodes: List[CarlaEpisodeSummary] = []

        for raw in iter_carla_route_records(RUN_DIR):
            rec = normalize_carla_route_record(RUN_DIR, test_type=TEST_TYPE, raw=raw)
            if rec is None:
                continue
            route_rows.append(rec)
            episodes.append(CarlaEpisodeSummary(
                driving_score=float(rec["driving_score"]),
                blocked=bool(rec["blocked"]),
            ))

        summary = []
        if episodes:
            s = carla_ds_active(episodes)
            summary.append({
                "sim": "carla",
                "map": MAP_NAME,
                "test_type": TEST_TYPE,
                "model": RUN_DIR.name,
                "ds_active_mean": s["ds_active_mean"],
                "ds_active_p50": s["ds_active_p50"],
                "ds_active_p95": s["ds_active_p95"],
                "blocked_rate": s["blocked_rate"],
                "n_total": s["n_total"],
                "n_active": s["n_active"],
            })

        write_csv(OUT_DIR / "route_metrics.csv", route_rows)
        write_csv(OUT_DIR / "summary_table.csv", summary)
        print(f"[OK] wrote: {OUT_DIR.resolve()}")
        return

    raise ValueError(f"Unknown SIM={SIM!r}. Use 'udacity' or 'carla'.")


if __name__ == "__main__":
    main()
