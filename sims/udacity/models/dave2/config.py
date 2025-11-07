MAP_NAME = "jungle"   # jungle or genroads
MODEL_NAME = "dave2"
INPUTS_GLOB = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/jungle/pid_20251107-160612/*.jpg"

INPUT_SHAPE = (120, 160, 3) # (row, col, ch)
NUM_OUTPUTS = 2
LEARNING_RATE = 1e-4
ALPHA_STEER = 0.7
VAL_SPLIT = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 1000
PATIENCE = 25
AUGMENTATIONS = []

