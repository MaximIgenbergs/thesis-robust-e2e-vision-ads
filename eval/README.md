# Evaluation Framework

This directory contains evaluation scripts for testing model robustness and generalization across different simulators and maps.

## Directory Structure

```
eval/
├── __init__.py
├── carla/                    # CARLA simulator evaluation
│   ├── cfg_generalization.yaml
│   ├── cfg_robustness.yaml
│   ├── run_generalization.py
│   └── run_robustness.py
├── genroads/                 # GenRoads (procedural roads) evaluation
│   ├── cfg_generalization.yaml
│   ├── cfg_robustness.yaml
│   ├── run_generalization.py
│   └── run_robustness.py
└── jungle/                   # Jungle map evaluation
    ├── cfg_generalization.yaml
    ├── cfg_robustness.yaml
    ├── run_generalization.py
    └── run_robustness.py
```

## Usage

### Jungle Map

```bash
python -m eval.jungle.run_robustness --model dave2

python -m eval.jungle.run_generalization --model dave2
```

### GenRoads

```bash
python -m eval.genroads.run_robustness --model dave2

python -m eval.genroads.run_generalization --model vit
```

### CARLA

```bash
python -m eval.carla.run_robustness --model tcp

python -m eval.carla.run_generalization --model tcp
```

## Configuration

Each evaluation type has a YAML configuration file:

## Output Structure

Evaluation runs are logged to:

```
runs/{map}/{test_type}/{model}_{timestamp}/
├── run_meta.json          # Run metadata
├── configs/               # Snapshot of configurations
├── env.txt                # pip freeze output
└── episodes/
    ├── ep_001/
    │   ├── meta.json      # Episode metadata
    │   ├── log.json       # Trajectory and action logs
    │   └── events.json    # Off-track/collision events
    ├── ep_002/
    └── ...
```

## CARLA-Specific Notes

CARLA evaluation uses the Leaderboard framework:

1. Start CARLA server: `./CarlaUE4.sh -quality-level=Low -carla-rpc-port=62100`
2. Configure `cfg_robustness.yaml` or `cfg_generalization.yaml`
3. Run evaluation script

Routes are defined in XML files under `scripts/carla/routes/`.

## Metrics

After evaluation, compute metrics using:

```bash
python -m metrics.compute_metrics
```

See [metrics/README.md](../metrics/README.md) for details.
