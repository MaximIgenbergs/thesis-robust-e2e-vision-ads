MAP_NAME = "genroads"   # jungle or genroads
MODEL_NAME = "dave2"
INPUTS_GLOB = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/genroads/train_dataset_gru/*.jpg"

INPUT_SHAPE = (120, 160, 3) # (row, col, ch)
NUM_OUTPUTS = 2
LEARNING_RATE = 1e-4
ALPHA_STEER = 0.7
VAL_SPLIT = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 1000
PATIENCE = 10
AUGMENTATIONS = []

