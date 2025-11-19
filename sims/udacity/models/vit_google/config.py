MAP_NAME    = "jungle"   # jungle or genroads
MODEL_NAME  = "vit"

# Same style as DAVE2: glob over Udacity JPEGs, paired with record_XXX.json
INPUTS_GLOB = "/home/maximigenbergs/thesis-robust-e2e-vision-ads/data/udacity/jungle/pid_20251029-174507/*.jpg"

# ViT input size (we resize inside the dataset)
IMG_SIZE = (224, 224)   # H, W

NUM_OUTPUTS  = 2            # [steering, throttle]
LEARNING_RATE = 3e-5        # smaller LR for fine-tuning pretrained ViT
ALPHA_STEER   = 0.7         # weight on steering in weighted MSE
VAL_SPLIT     = 0.20
RANDOM_SEED   = 42
BATCH_SIZE    = 64
EPOCHS        = 500
PATIENCE      = 30
NUM_WORKERS   = 4           # DataLoader workers

# ViT backbone from timm, pretrained on ImageNet-21k + AugReg, representative ViT-B/16
BACKBONE_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"
