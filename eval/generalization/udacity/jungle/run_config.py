from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[5]
from sims.udacity.maps.configs.paths import CKPTS_DIR

# Simulator:
SHOW_IMAGE = True # simulator live preview
STEPS = 2000 # max steps per episode
SAVE_IMAGES = False
IMAGE_SIZE = (240, 320) # (H, W) Dave2: (240, 320)
HOST = "127.0.0.1"
PORT = 9091

# Directories
SIM = REPO_ROOT / "sims/udacity/binaries/udacity_linux_reverse/udacity.x86_64" 
RUNS_DIR = REPO_ROOT / "runs" / "udacity" / "jungle"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_ROOT / "data" / "udacity" / "jungle"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints
DAVE2_CKPT = ""
DAVE2_GRU_CKPT = CKPTS_DIR / "dave2_gru/jungle_20251103-150739/best_model.h5"

# 1) Select map to evaluate on
MAP_NAME = "jungle_reverse" # "jungle", "jungle_reverse"

# 2) Select start waypoint (on selected map)
START_WAYPOINT = 200 # 0 to 999

# 3) Select model to evaluate
MODEL_NAME = "dave2_gru" # "dave2", "dave2_gru"

