from pathlib import Path
import platform

REPO_ROOT = Path(__file__).resolve().parents[5]
BINARES_DIR = REPO_ROOT / "sims" / "udacity" / "maps" / "genroads" / "binaries"

from sims.udacity.maps.configs.paths import CKPTS_DIR

system = platform.system()
machine = platform.machine()

if system == "Darwin":
    SIM = BINARES_DIR / "udacity_macos/udacity_sim_weather_sky_ready_angles_fortuna.app" 
elif system == "Linux":
    SIM = BINARES_DIR / "udacity_linux/udacity_binary.x86_64"
else:
    raise RuntimeError(f"Unsupported platform: {system}")

RUNS_DIR = REPO_ROOT / "runs" / "udacity" / "genroads"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_ROOT / "data" / "udacity" / "genroads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints
DAVE2_CKPT = CKPTS_DIR / "dave2/genroads_20251028-145557/best_model.h5"
DAVE2_GRU_CKPT = CKPTS_DIR / "dave2_gru/genroads_20251028-142517/best_model.h5"
DAVE2_PD_CKPT = REPO_ROOT / "external/perturbation-drive/examples/models/checkpoints/dave_90k_v1.h5"