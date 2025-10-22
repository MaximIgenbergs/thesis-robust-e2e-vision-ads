# Validation & data
VAL_SPLIT   = 0.10
RANDOM_SEED = 42

# Per-frame input (H, W, C)
INPUT_SHAPE = (66, 200, 3)

# Temporal settings
SEQ_LEN     = 5          # number of frames per sample
SEQ_STRIDE  = 1          # slide by this many frames between sequences
LABEL_FROM  = "last"     # "last" or "center"

# Model/optimization
GRU_UNITS     = 128
ALPHA_STEER   = 0.6
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64        # usually smaller than pure-CNN because of sequences
EPOCHS        = 30
PATIENCE      = 4
AUGMENTATIONS = {
    "brightness": False,
    "flip": False,
}
