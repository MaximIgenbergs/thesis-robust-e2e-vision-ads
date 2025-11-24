MODEL_NAME = "dave2_gru"

# jungle
# MAP_NAME = "jungle"
# DATA_DIR = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/jungle/pid_20251107-160612"

# genroads
MAP_NAME = "genroads"
DATA_DIR = "/Users/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/genroads/train_dataset_gru"

INPUT_SHAPE = (120, 160, 3) # (H, W, C)
NUM_OUTPUTS = 2
LEARNING_RATE = 1e-4
ALPHA_STEER = 0.7
VAL_SPLIT = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 1000
PATIENCE = 25
AUGMENTATIONS = []

SEQ_LEN = 3
STRIDE = 1
