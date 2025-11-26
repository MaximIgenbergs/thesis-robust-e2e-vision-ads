from pathlib import Path
import platform

REPO_ROOT = Path(__file__).resolve().parents[5]
BINARES_DIR = REPO_ROOT / "scripts" / "udacity" / "maps" / "jungle" / "binaries"

from scripts import CKPTS_DIR

system = platform.system()
machine = platform.machine()

if system == "Darwin":
    if machine == "arm64":
        SIM = BINARES_DIR / "udacity_mac_silicon/udacity.app"
    else:
        SIM = BINARES_DIR / "udacity_mac_intel-64-bit/udacity.app"
elif system == "Linux":
    SIM = BINARES_DIR / "udacity_linux/udacity.x86_64"
else:
    raise RuntimeError(f"Unsupported platform: {system} on {machine}")

RUNS_DIR = REPO_ROOT / "runs" / "udacity" / "jungle"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_ROOT / "data" / "udacity" / "jungle"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints
DAVE2_CKPT = CKPTS_DIR / "dave2/jungle_20251107-173326/best_model.h5"
DAVE2_GRU_CKPT = CKPTS_DIR / "dave2_gru/jungle_20251103-150739/best_model.h5"
VIT_CKPT = CKPTS_DIR / "vit/jungle_20251119-124028/best_model.ckpt"