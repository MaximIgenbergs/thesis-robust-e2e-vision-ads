from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]

from sims.udacity.maps.configs.paths import CKPTS_DIR

SIM = REPO_ROOT / "sims/udacity/binaries/udacity_linux/udacity.x86_64" 
RUNS_DIR = REPO_ROOT / "runs" / "udacity" / "jungle"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_ROOT / "data" / "udacity" / "jungle"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints
DAVE2_CKPT = ""
DAVE2_GRU_CKPT = CKPTS_DIR / "dave2_gru/jungle_20251103-150739/best_model.h5"
