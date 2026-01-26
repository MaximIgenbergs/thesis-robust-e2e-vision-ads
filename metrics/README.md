# Metrics Computation

Computes results and tables for the thesis.

## Configuration

Set environment variables: `RUNS_ROOT`, `RESULTS_ROOT`

## Special Cases & Hardcoded Patches

- Hardcoded tiny routes path: `/media/maxim/Elements/maximigenbergs/runs/carla/generalization/tcp/20251231_045136/tcp/`
- `genroads_robustness_dave2_gru` uses `paired_severities=True`
- `paired_severities=True`: For Udacity runs with `reconnect=false`. Changes log priority: `new_log.json` → `pd_log.json` → `log.json`
- `new_log.json` contains reconstructed PID actions from `pd_log.json` (for analysis when PID actions missing from original logs)

## Usage

```bash
python -m metrics.compute_metrics
```