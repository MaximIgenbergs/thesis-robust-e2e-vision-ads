MAP_NAME = "genroads"   # or "jungle" if training on that domain
INPUTS_GLOB = "data/udacity/genroads/train_dataset/*.jpg"

INPUT_SHAPE = (120, 160, 3) # (row, col, ch)
NUM_OUTPUTS = 2
LEARNING_RATE = 1e-4
ALPHA_STEER = 0.7
VAL_SPLIT = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
AUGMENTATIONS = []

