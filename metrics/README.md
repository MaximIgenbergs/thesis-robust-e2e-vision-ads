# Metrics Computation

This module computes evaluation metrics and generates result tables for the thesis.

## Directory Structure

```
metrics/
├── __init__.py
├── aggregate.py           # Aggregation functions for summaries
├── compute_episode.py     # Episode-level metric computation
├── compute_metrics.py     # Main entry point
├── constants.py           # Metric constants and thresholds
├── failure_udacity.py     # Failure detection for Udacity
├── generate_report_tables.py  # LaTeX table generation
├── io_carla.py            # CARLA log parsing
├── io_udacity.py          # Udacity log parsing
├── metrics.py             # Core metric functions
├── plots.py               # Visualization utilities
├── reconstruct_pid_actions.py  # PID action reconstruction
├── report_tables.py       # Table formatting
└── utils.py               # Utility functions
```

## Configuration

Set environment variables before running:

```bash
export RUNS_ROOT=/path/to/runs      # Directory containing evaluation runs
export RESULTS_ROOT=results          # Output directory for computed metrics
```

## Usage

```bash
python -m metrics.compute_metrics
```

## Metrics Computed

### Tracking Metrics

| Metric           | Description                            |
| ---------------- | -------------------------------------- |
| `xte_abs_mean`   | Mean absolute cross-track error        |
| `xte_abs_p95`    | 95th percentile absolute XTE           |
| `angle_abs_mean` | Mean absolute heading error            |
| `angle_abs_p95`  | 95th percentile absolute heading error |

### Action Deviation Metrics

| Metric             | Description                        |
| ------------------ | ---------------------------------- |
| `pid_dev_mean`     | Mean deviation from PID reference  |
| `pid_dev_p95`      | 95th percentile deviation from PID |
| `pid_mae_steer`    | MAE of steering vs PID             |
| `pid_mae_throttle` | MAE of throttle vs PID             |

### Robustness Metrics

| Metric             | Description                                |
| ------------------ | ------------------------------------------ |
| `corruption_error` | Performance drop under perturbation        |
| `mCE`              | Mean corruption error across perturbations |
| `relative_drop`    | Relative performance decrease              |
| `auc_severity`     | Area under severity curve                  |

### CARLA-Specific Metrics

| Metric          | Description                                |
| --------------- | ------------------------------------------ |
| `driving_score` | CARLA Leaderboard driving score            |
| `blocked_rate`  | Fraction of episodes where agent got stuck |
| `time_to_block` | Time until agent blocked                   |

## Special Cases

- `paired_severities=True`: For runs with `reconnect=false`, changes log file priority
- `new_log.json` contains reconstructed PID actions when missing from original logs

## Output

Results are written to `RESULTS_ROOT`:

```
results/{map}/{test_type}/{model}_{timestamp}/
├── entry_metrics.csv      # Per-episode metrics
├── summary.csv            # Aggregated metrics
├── robustness_summary.csv # Robustness-specific metrics
└── tables/                # LaTeX tables for thesis
```

## Core Functions

### `metrics.py`

```python
from metrics.metrics import (
    summarize_abs,           # Compute mean/percentile of absolute values
    carla_blocked_rate,      # CARLA blocking rate
    corruption_error,        # CE computation
    mean_corruption_error,   # mCE computation
    relative_drop,           # Relative performance drop
    auc_over_severity,       # AUC over severity levels
)
```

### `aggregate.py`

```python
from metrics.aggregate import (
    aggregate_udacity_summary,   # Aggregate Udacity metrics
    aggregate_carla_summary,     # Aggregate CARLA metrics
    robustness_summaries,        # Robustness analysis
    mce_over_all_corruptions,    # mCE across all perturbations
)
```