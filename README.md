# Dual-Axis Testing of Visual Robustness and Topological Generalization in Vision-based End-to-End Driving Models

Bachelor Thesis project evaluating the robustness and generalization capabilities of end-to-end autonomous driving models across multiple simulators and test scenarios.

## Overview

This repository implements a comprehensive evaluation framework for testing vision-based end-to-end driving models along two axes:

1. **Visual Robustness**: Testing model performance under visual perturbations (weather, color, tone, blur, noise, distortions, transformations, overlays)
2. **Topological Generalization**: Testing model performance on unseen topologies and scenarios

## Repository Structure

```
├── checkpoints/           # model weights
│   ├── dave2/
│   ├── dave2_gru/
│   └── vit/
│   └── tcp/
├── data/                  # Training datasets
│   ├── genroads/
│   ├── jungle/
├── eval/                  # Evaluation scripts
│   ├── carla/            # CARLA simulator evaluation
│   ├── genroads/         # GenRoads evaluation
│   └── jungle/           # Jungle map evaluation
├── external/              # Git submodules
│   ├── InterFuser/       # InterFuser agent for CARLA
│   ├── TCP/              # TCP agent for CARLA
│   ├── perturbation-drive/ # Perturbation injection framework
│   └── udacity_gym/      # Udacity simulator interface for jungle map
├── metrics/               # Metrics computation and aggregation
├── results/               # Computed metrics and tables
├── runs/                  # Evaluation run logs
├── scripts/               # Training and data collection
│   ├── carla/            # CARLA-specific scripts
│   └── udacity/          # Udacity-specific scripts
│       ├── adapters/     # Model inference adapters
│       ├── logging/      # Run logging utilities
│       ├── maps/         # Map-specific scripts
│       └── models/       # Model definitions and training
└── binaries/              # Simulator executables (not tracked)
```

## Setup


### 1. Clone Repository

```bash
git clone --recursive https://github.com/MaximIgenbergs/thesis-robust-e2e-vision-ads.git
cd thesis-robust-e2e-vision-ads
```

### 2. Download Pre-trained Models and Data from Hugging Face

All models, datasets, and evaluation runs are hosted on Hugging Face:

**Models:**
- [dave2](https://huggingface.co/maxim-igenbergs/dave2) → `checkpoints/dave2/`
- [dave2-gru](https://huggingface.co/maxim-igenbergs/dave2-gru) → `checkpoints/dave2_gru/`
- [vit](https://huggingface.co/maxim-igenbergs/vit) → `checkpoints/vit/`
- [carla-tcp-repro](https://huggingface.co/maxim-igenbergs/carla-tcp-repro) → `checkpoints/tcp/`

**Datasets:**
- [thesis-data](https://huggingface.co/datasets/maxim-igenbergs/thesis-data) → `data/`
- [thesis-runs](https://huggingface.co/datasets/maxim-igenbergs/thesis-runs) → `runs/`

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download models
huggingface-cli download maxim-igenbergs/dave2 --local-dir checkpoints/dave2
huggingface-cli download maxim-igenbergs/dave2-gru --local-dir checkpoints/dave2_gru
huggingface-cli download maxim-igenbergs/vit --local-dir checkpoints/vit
huggingface-cli download maxim-igenbergs/carla-tcp-repro --local-dir checkpoints/tcp

# Download datasets
huggingface-cli download maxim-igenbergs/thesis-data --repo-type dataset --local-dir data
huggingface-cli download maxim-igenbergs/thesis-runs --repo-type dataset --local-dir runs
```

### 3. Environment Setup

Choose the appropriate setup script based on your platform and target simulator:

**GenRoads (Linux):**
```bash
./setup_venv_genroads_linux.sh
source envs/.venv-genroads/bin/activate
```

**Jungle (Linux):**
```bash
./setup_venv_jungle_linux.sh
source envs/.venv-udc/bin/activate
```

### 4. Initialize Git Submodules

```bash
git submodule update --init --recursive
```

## Usage

### Training Models

Train models on collected driving data:

```bash
# Train DAVE2
python -m scripts.udacity.models.dave2.train

# Train DAVE2-GRU
python -m scripts.udacity.models.dave2_gru.train

# Train ViT
python -m scripts.udacity.models.vit.train
```

Configure training parameters in `scripts/udacity/models/{model}/config.py`.

### Data Collection

Collect training data using a PD-controller:

```bash
# Jungle map
python -m scripts.udacity.maps.jungle.run_data_collection

# GenRoads
python -m scripts.udacity.maps.genroads.run_data_collection
```

### Evaluation

#### Robustness Evaluation

Test models under image perturbations:

```bash

python -m eval.jungle.run_robustness --model dave2

python -m eval.genroads.run_robustness --model vit

python -m eval.carla.run_robustness --model tcp
```

#### Generalization Evaluation

Test models on unseen road layouts:

```bash
python -m eval.jungle.run_generalization --model dave2_gru

python -m eval.genroads.run_generalization --model vit

python -m eval.carla.run_generalization --model tcp
```

### Metrics Computation

Compute metrics from evaluation runs:

```bash
# Set paths
export RUNS_ROOT=/path/to/runs
export RESULTS_ROOT=results

# Compute all metrics
python -m metrics.compute_metrics
```

See [metrics/README.md](metrics/README.md) for detailed metrics documentation.

## External Dependencies

This project uses the following external repositories as git submodules:

- [PerturbationDrive](https://github.com/MaximIgenbergs/perturbation-drive) - Image perturbation framework
- [TCP](https://github.com/OpenDriveLab/TCP) - Trajectory-guided Control Prediction
- [InterFuser](https://github.com/opendilab/InterFuser) - Multi-modal fusion agent

## License

This project is part of a Bachelor Thesis at the Technical University of Munich and fortiss.

## Citation

If you use this work, please cite:

```bibtex
@thesis{igenbergs2026dualaxis,
  title={Dual-Axis Testing of Visual Robustness and Topological Generalization in Vision-based End-to-End Driving Models},
  author={Igenbergs, Maxim},
  year={2026},
  type={Bachelor Thesis}
}
```
