MODEL_NAME = "vit"

# jungle
# MAP_NAME = "jungle"
# DATA_DIR = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/jungle/pid_20251107-160612"

# genroads
MAP_NAME = "genroads"
DATA_DIR = "/Users/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/genroads/train_dataset_gru" # "/Users/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/genroads/pid_20251107-160612"

INPUT_SHAPE = (160, 160, 3)  # (H, W, C), matches VisionTransformer(image_size=160)
NUM_OUTPUTS = 2
LEARNING_RATE = 2e-4
VAL_SPLIT = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 64          # adjust as needed for GPU vs Mac
EPOCHS = 2
PATIENCE = 20
AUGMENTATIONS = ["augmix", "flip"]
