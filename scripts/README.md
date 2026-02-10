# Scripts

This directory contains all training, data collection, and utility scripts for the Udacity and CARLA simulators.

## Directory Structure

```
scripts/
├── __init__.py              # Path utilities (abs_path, load_cfg)
├── carla/                   # CARLA-specific scripts
│   ├── agents/              # Custom CARLA agents
│   │   └── perturbation_agent.py  # TCP agent with perturbation support
│   └── routes/              # Route definitions
│       └── routes_robustness.xml
└── udacity/                 # Udacity simulator scripts
    ├── adapters/            # Model inference adapters
    ├── configs/             # Configuration files
    ├── logging/             # Run logging utilities
    ├── maps/                # Map-specific scripts
    └── models/              # Model definitions and training
```

## Utility Functions

The `scripts/__init__.py` module provides:

```python
from scripts import abs_path, load_cfg, ROOT, CKPTS_DIR

# Get absolute path relative to repo root
path = abs_path("checkpoints/dave2/best_model.h5")

# Load YAML configuration
cfg = load_cfg("eval/jungle/cfg_robustness.yaml")
```

## Udacity Scripts

### Model Adapters (`udacity/adapters/`)

- `dave2_adapter.py`
- `dave2_gru_adapter.py`
- `vit_adapter.py`

Usage:
```python
from scripts.udacity.adapters.utils.build_adapter import build_adapter

adapter, ckpt_path = build_adapter("dave2", model_config)
prediction = adapter.predict(image)
```

### Model Training (`udacity/models/`)

Each model has its own subdirectory with:
- `config.py` - Hyperparameters and paths
- `model.py` - Model architecture definition
- `train.py` - Training script
- `utils/` - Data loading utilities

**DAVE2:**
```bash
python -m scripts.udacity.models.dave2.train
```

**DAVE2-GRU:**
```bash
python -m scripts.udacity.models.dave2_gru.train
```

**ViT:**
```bash
python -m scripts.udacity.models.vit.train
```

### Data Collection (`udacity/maps/`)

Collect training data using PD-controller:

**Jungle Map:**
```bash
python -m scripts.udacity.maps.jungle.run_data_collection
```

**GenRoads:**
```bash
python -m scripts.udacity.maps.genroads.run_data_collection
```

Data is saved as image/JSON pairs:
```
data/{map}/pid_YYYYMMDD-HHMMSS/
├── image_000001.jpg
├── record_000001.json
├── image_000002.jpg
├── record_000002.json
└── ...
```

### Logging Utilities (`udacity/logging/`)

- `data_collection_runs.py` - Data collection run management
- `training_runs.py` - Training run logging
- `eval_runs.py` - Evaluation run logging
- `data_conversion.py` - Data format conversion

## CARLA Scripts

### Perturbation Agent (`carla/agents/perturbation_agent.py`)

Custom TCP agent that supports image perturbations during CARLA evaluation.

### Routes (`carla/routes/`)

XML route definitions for CARLA evaluation scenarios.

## Configuration

Model configurations are in `udacity/models/{model}/config.py`:
