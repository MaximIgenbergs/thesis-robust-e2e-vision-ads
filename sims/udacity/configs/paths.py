from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPTS_DIR = REPO_ROOT / "sims/udacity/checkpoints"
PD_SIM = REPO_ROOT / "external/perturbation-drive/examples/udacity/sim/udacity_linux/udacity_binary.x86_64"
SIM = REPO_ROOT / "sims/udacity/binaries/udacity/udacity.x86_64"


# Checkpoints
DAVE2_CKPT = CKPTS_DIR / "dave2_pd/training_runs/dave2_run_2025-10-22_17-16-23/best_model.h5"
DAVE2_GRU_CKPT = CKPTS_DIR / "dave2_gru/training_runs/dave2_gru_run_2025-10-22_15-13-58/best_model.h5"

DAVE2_PD_CKPT = REPO_ROOT / "external/perturbation-drive/examples/models/checkpoints/dave_90k_v1.h5"

RUNS_DIR = REPO_ROOT / "runs" / "udacity"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
