# This file represent configuration settings and constants in the project
import pathlib
import sys
from collections import defaultdict
import multiprocessing
multiprocessing.set_start_method("fork", force=True) # This is required for the simulator to work on MacOS, maybe has to be removed for linux

# Paths

PROJECT_DIR = pathlib.Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_DIR.joinpath("model/ckpts")
LOG_DIR = PROJECT_DIR.joinpath("Logs")

# Simulator settings
simulator_infos = defaultdict(dict)

# pick the right executable name
if sys.platform == "darwin":
    exe = PROJECT_DIR / "simulator" / "builds_mac.app"
elif sys.platform.startswith("linux"):
    exe = PROJECT_DIR / "simulator" / "build_angle_diff" / "udacity.x86_64"
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


# Device settings
ACCELERATOR = "gpu"  # choose between gpu or cpu
DEVICE = 0  # if multiple gpus are available
DEFAULT_DEVICE = f'cuda:{DEVICE}' if ACCELERATOR == 'gpu' else 'cpu'

# Simulator settings
simulator_infos = defaultdict(dict)
simulator_infos[1]["exe_path"] = exe
simulator_infos[1]['host'] = "127.0.0.1"
simulator_infos[1]['port'] = 4567

# Training settings
Training_Configs = defaultdict()
Training_Configs['training_data_dir'] = PROJECT_DIR.joinpath("Data")
Training_Configs['TEST_SIZE'] = 0.2  # split of training data used for the validation set (keep it low)
Training_Configs['BATCH_SIZE'] = 128
Training_Configs['WITH_BASE'] = False
Training_Configs['BASE_MODEL'] = 'track1-steer-throttle.h5'
Training_Configs['LEARNING_RATE'] = 1e-4
Training_Configs['EPOCHS'] = 200
Training_Configs['SHUFFLE_DATA'] = True
# SAMPLE_DATA = False
# AUG_CHOOSE_IMAGE = True
Training_Configs['AUG'] = defaultdict()
# Training_Configs['AUG']['ENABLE'] = True # always enable
Training_Configs['AUG']['USE_LEFT_RIGHT'] = True
Training_Configs['AUG']['RANDOM_FLIP'] = True
Training_Configs['AUG']['RANDOM_TRANSLATE'] = True
Training_Configs['AUG']['RANDOM_SHADOW'] = True
Training_Configs['AUG']['RANDOM_BRIGHTNESS'] = True

# Track settings
Track_Infos = defaultdict(dict)
Track_Infos[1]['track_name'] = 'lake'
Track_Infos[1]['model_path'] = CHECKPOINT_DIR.joinpath('ads', 'track1-dave2-168.h5')
Track_Infos[1]['simulator'] = simulator_infos[1]
Track_Infos[1]['driving_style'] = ["normal_lowspeed", "reverse_lowspeed", "normal_lowspeed", "reverse_lowspeed"]
Track_Infos[1]['training_data_dir'] = Training_Configs['training_data_dir'].joinpath('lane_keeping_data', 'track1_throttle')

Track_Infos[2]['track_name'] = 'jungle'
Track_Infos[2]['model_path'] = CHECKPOINT_DIR.joinpath('ads', 'track3-dave2-191.h5') # TODO: add the right model
Track_Infos[2]['simulator'] = simulator_infos[1]
Track_Infos[2]['driving_style'] = ["normal_lowspeed", "reverse_lowspeed", "normal_lowspeed", "reverse_lowspeed"]
Track_Infos[2]['training_data_dir'] = Training_Configs['training_data_dir'].joinpath('lane_keeping_data', 'track2_throttle')

Track_Infos[3]['track_name'] = 'mountain'
Track_Infos[3]['model_path'] = CHECKPOINT_DIR.joinpath('ads', 'track3-dave2-191.h5')
Track_Infos[3]['simulator'] = simulator_infos[1]
Track_Infos[3]['driving_style']  = ["normal", "reverse", "normal", "reverse"]
Track_Infos[3]['training_data_dir'] = Training_Configs['training_data_dir'].joinpath('lane_keeping_data', 'track3_throttle')
# TODO: add code to override default settings

model_cfgs = defaultdict()
model_cfgs['image_width'] = 320
model_cfgs['image_height'] = 160
model_cfgs['image_depth'] = 3
model_cfgs['resized_image_width'] = 160
model_cfgs['resized_image_height'] = 80

model_cfgs['num_outputs'] = 2 # when we wish to predict steering and throttle:
