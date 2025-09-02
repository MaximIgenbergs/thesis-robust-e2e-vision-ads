# Configuration settings for Python client
import pathlib
import sys
from collections import defaultdict
import torch

# Add project root to PYTHONPATH so shared utils can be imported
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Drive Script Configuration
TRACK = "city"  # lake, jungle, mountain, city, generator
DAYTIME = "day"
WEATHER = "sunny"
ENABLE_LOGGING = False
STEPS = 7000
EGO_CIRCUIT_NAME = "MainWayPointCircuit"
EGO_SPAWN_INDEX = {
    "jungle": 44,
    "city": 350,
    "lake": 1,
    "mountain": 1,
    "generator": 1
}.get(TRACK, 1)
RANDOM_NPCS = 0 # Number of randomly spawned NPC vehicles
FIXED_NPCS = [ # Each entry describes one NPC car to spawn with fixed settings
    # Example:
    # {
    #     "name": "npc1",
    #     "prefab": "Objects/CarRed",
    #     "circuit": "MainWayPointCircuit",
    #     "spawn_index": 10,
    #     "speed": 20.0
    # }
]

# Paths
CHECKPOINT_DIR = PROJECT_DIR.joinpath("model/ckpts")
LOG_DIR = PROJECT_DIR.joinpath("logs")
TRAINING_DIR = PROJECT_DIR.joinpath("Data/log_pid_02_06_2025_13_01_12")

# Simulator settings
simulator_infos = defaultdict(dict)

if sys.platform == "darwin":
    exe = PROJECT_DIR / "udacity_gym_binaries" / "builds_mac.app"
elif sys.platform.startswith("linux"):
    exe = PROJECT_DIR / "udacity_gym_binaries" / "build_angle_diff" / "udacity.x86_64"
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

# Device settings
if torch.cuda.is_available():
    ACCELERATOR = "gpu" # 4090
    DEVICE = 0
    DEFAULT_DEVICE = f"cuda:{DEVICE}"
elif torch.backends.mps.is_available():
    ACCELERATOR = "mps" # Apple Silicon GPU
    DEVICE = 0
    DEFAULT_DEVICE = "mps"
else:
    ACCELERATOR = "cpu"
    DEVICE = 0
    DEFAULT_DEVICE = "cpu"

# Network ports for the simulator
simulator_infos[1]['exe_path'] = exe
simulator_infos[1]['host'] = "127.0.0.1"
simulator_infos[1]['cmd_port'] = 55001
simulator_infos[1]['telemetry_port'] = 56001
simulator_infos[1]['event_port'] = 57001
simulator_infos[1]['others_port'] = 58001

# Legacy:

# # Training settings
# Training_Configs = defaultdict()
# Training_Configs['training_data_dir'] = PROJECT_DIR.joinpath("Data")
# Training_Configs['TEST_SIZE'] = 0.2
# Training_Configs['BATCH_SIZE'] = 128
# Training_Configs['WITH_BASE'] = False
# Training_Configs['BASE_MODEL'] = 'track1-steer-throttle.h5'
# Training_Configs['LEARNING_RATE'] = 1e-4
# Training_Configs['EPOCHS'] = 200
# Training_Configs['SHUFFLE_DATA'] = True
# Training_Configs['AUG'] = defaultdict()
# Training_Configs['AUG']['USE_LEFT_RIGHT'] = True
# Training_Configs['AUG']['RANDOM_FLIP'] = True
# Training_Configs['AUG']['RANDOM_TRANSLATE'] = True
# Training_Configs['AUG']['RANDOM_SHADOW'] = True
# Training_Configs['AUG']['RANDOM_BRIGHTNESS'] = True

# # Model configuration
# model_cfgs = defaultdict()
# model_cfgs['image_width'] = 320
# model_cfgs['image_height'] = 160
# model_cfgs['image_depth'] = 3
# model_cfgs['resized_image_width'] = 160
# model_cfgs['resized_image_height'] = 80
# model_cfgs['num_outputs'] = 2
