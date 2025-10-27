from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

from sims.udacity.configs.paths import CKPTS_DIR

SIM = REPO_ROOT / "sims/udacity/binaries/udacity/udacity.x86_64"
RUNS_DIR = REPO_ROOT / "runs" / "udacity" / "jungle"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints
DAVE2_CKPT = CKPTS_DIR / "dave2/jungle_*/best_model.h5"
DAVE2_GRU_CKPT = CKPTS_DIR / "dave2_gru/jungle_*/best_model.h5"


