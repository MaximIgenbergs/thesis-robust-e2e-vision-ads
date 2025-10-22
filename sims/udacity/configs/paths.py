from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PD_SIM = REPO_ROOT / "external/perturbation-drive/examples/udacity/sim/udacity/udacity_sim_weather_sky_ready_angles_fortuna.app"

# Checkpoints
DAVE2_CKPT = REPO_ROOT / "external/perturbation-drive/examples/models/checkpoints/dave_90k_v1.h5" # "sims/udacity/checkpoints/dave2/training_runs/run_2025-06-09_17-58-46/best_model.h5"
DAVE2_GRU_CKPT = REPO_ROOT / "sims/udacity/checkpoints/dave2_gru/training_runs/run_XX/best_model.h5"
VIT_CKPT = REPO_ROOT / "sims/udacity/checkpoints/vit/training_runs/run_XX/best_model.h5"

# Training Data
# DATA_ROOT = REPO_ROOT / "data" / "collections"
# TRAIN_COLLECTION_NAME = "pid_20250602T130112"
# TRAIN_DATA_DIR = DATA_ROOT / TRAIN_COLLECTION_NAME
# TRAIN_IMG_DIR  = TRAIN_DATA_DIR / "image"
# TRAIN_LOG_PATH = TRAIN_DATA_DIR / "log.csv"

# fail early if the selected collection is missing
# if not TRAIN_DATA_DIR.exists():
#     raise FileNotFoundError(
#         f"Training data directory not found: {TRAIN_DATA_DIR}\n"
#         "Edit TRAIN_COLLECTION_NAME in sims/udacity/configs/paths.py."
#     )

TRAINING_RUNS_DIR = REPO_ROOT / "sims" / "udacity" / "checkpoints" / "dave2" / "training_runs"
TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)

RUNS_DIR = REPO_ROOT / "runs" / "udacity"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
