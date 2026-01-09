MODEL_NAME = "dave2_gru"

# jungle
# MAP_NAME = "jungle"
# DATA_DIR = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/jungle/pid_20251029-174507"

# genroads
MAP_NAME = "genroads"
DATA_DIR = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/genroads/pid_20251201-163211"

INPUT_SHAPE = (120, 160, 3) # (H, W, C)
NUM_OUTPUTS = 2
LEARNING_RATE = 1e-4
ALPHA_STEER = 0.7
VAL_SPLIT = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 2000
PATIENCE = 32
AUGMENTATIONS = []

SEQ_LEN = 3
STRIDE = 1 # frames between sequences
