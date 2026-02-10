# External Dependencies

This directory contains git submodules for external libraries and frameworks used in this project.

## Submodules

### PerturbationDrive

**Repository:** [MaximIgenbergs/perturbation-drive](https://github.com/MaximIgenbergs/perturbation-drive) (Replication branch)

### TCP (Trajectory-guided Control Prediction)

**Repository:** [OpenDriveLab/TCP](https://github.com/OpenDriveLab/TCP)

### InterFuser

**Repository:** [opendilab/InterFuser](https://github.com/opendilab/InterFuser)

### Udacity Gym

Custom interface for the Udacity self-driving car simulator.

## Initialization

Initialize all submodules after cloning:

```bash
git submodule update --init --recursive
```

## Installation

Each submodule has its own dependencies. The setup scripts handle installation:

```bash
# For Udacity experiments
./setup_venv_genroads_linux.sh
# and
./setup_venv_jungle_linux.sh
```

For CARLA experiments (TCP/InterFuser), follow TCP/InterFuser installation guides

## Directory Structure

```
external/
├── __init__.py
├── InterFuser/           # InterFuser CARLA agent
├── TCP/                  # TCP CARLA agent
├── carla-test/           # CARLA testing utilities
├── perturbation-drive/   # Image perturbation framework
└── udacity_gym/          # Udacity simulator interface
```