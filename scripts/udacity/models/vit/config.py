MODEL_NAME = "vit"

# jungle
# MAP_NAME = "jungle"
# DATA_DIR = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/jungle/pid_20251029-174507"

# genroads
MAP_NAME = "genroads"
DATA_DIR = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/TBD"

INPUT_SHAPE = (160, 160, 3)  # (H, W, C), matches VisionTransformer(image_size=160)
NUM_OUTPUTS = 2
LEARNING_RATE = 0.0002
VAL_SPLIT = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 256
EPOCHS = 2000
PATIENCE = 20
AUGMENTATIONS = ["augmix", "flip"]
