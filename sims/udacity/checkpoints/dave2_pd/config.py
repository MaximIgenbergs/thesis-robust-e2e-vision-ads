# Shapes & hyperparams for the baseline DAVE-2 trainer (no CLI)
# If sims.udacity.configs.paths.TRAINING_RUNS_DIR exists, trainers will save there.
INPUT_SHAPE = (120, 160, 3)      # (row, col, ch)
NUM_OUTPUTS = 2                   # 1 = steering only, 2 = steering+throttle

# Data
INPUTS_GLOB = "data/udacity/generated_roads/train_dataset_gru/*.jpg"
VAL_SPLIT   = 0.20
RANDOM_SEED = 42

# Optimizer / loss
LEARNING_RATE = 1e-4
ALPHA_STEER   = 0.7               # only used when NUM_OUTPUTS == 2 (weighted MSE)

# Training
BATCH_SIZE = 64
EPOCHS     = 50
PATIENCE   = 10

# Metadata (purely informational in meta.json)
AUGMENTATIONS = []

# Run directory layout (under repo_root/runs/<subdir> if TRAINING_RUNS_DIR not defined)
RUNS_SUBDIR      = "udacity/dave2"
RUN_NAME_PREFIX  = "dave2"
SAVE_BEST_FILENAME = "best_model.h5"
