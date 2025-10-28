from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]

from sims.udacity.maps.configs.paths import CKPTS_DIR

SIM = REPO_ROOT / "external/perturbation-drive/examples/udacity/sim/udacity_linux/udacity_binary.x86_64"
RUNS_DIR = REPO_ROOT / "runs" / "udacity" / "genroads"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints
DAVE2_CKPT = CKPTS_DIR / "dave2/genroads_20251028-145557/best_model.h5"
DAVE2_GRU_CKPT = CKPTS_DIR / "dave2_gru/genroads_20251028-142517/best_model.h5"
DAVE2_PD_CKPT = REPO_ROOT / "external/perturbation-drive/examples/models/checkpoints/dave_90k_v1.h5"