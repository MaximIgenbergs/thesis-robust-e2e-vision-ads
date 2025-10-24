SEVERITIES = [1, 2, 3, 4]
EPISODES = 1 # repeats per (perturb, severity) per road
RECONNECT = False # reconnect between severities for stability
IMAGE_SIZE = (240, 320) # (H, W) Dave2: (240, 320), ViT: TBD
SHOW_IMAGE = True # simulator live preview
HOST = "127.0.0.1"
PORT = 9091
MODEL_NAME = "dave2_gru" # "dave2", "dave2_gru"