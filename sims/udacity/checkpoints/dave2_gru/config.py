# Config for the GRU trainer (no CLI)
INPUT_SHAPE = (120, 160, 3)      # (row, col, ch)
NUM_OUTPUTS = 2                   # 1 or 2

# Data
INPUTS_GLOB = "data/udacity/generated_roads/train_dataset_gru/*.jpg"
VAL_SPLIT   = 0.20
RANDOM_SEED = 42

# Temporal settings
SEQ_LEN = 3
STRIDE  = 1

# Optimizer / loss
LEARNING_RATE = 1e-4
ALPHA_STEER   = 0.7              # only used when NUM_OUTPUTS == 2

# Training
BATCH_SIZE = 64
EPOCHS     = 50
PATIENCE   = 10

# Metadata
AUGMENTATIONS = []

# Run directory layout
RUNS_SUBDIR        = "udacity/dave2_gru"
RUN_NAME_PREFIX    = "dave2_gru"
SAVE_BEST_FILENAME = "best_model.h5"
