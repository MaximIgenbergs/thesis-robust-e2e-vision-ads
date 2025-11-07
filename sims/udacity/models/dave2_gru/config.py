MAP_NAME = "jungle"   # or "jungle" if training on that domain
MODEL_NAME = "dave2_gru"
INPUTS_GLOB = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/jungle/pid_20251107-160612/*.jpg" # "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/jungle/train_dataset_gru/*.jpg"

INPUT_SHAPE = (120, 160, 3) # (row, col, ch)
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
STRIDE  = 1
