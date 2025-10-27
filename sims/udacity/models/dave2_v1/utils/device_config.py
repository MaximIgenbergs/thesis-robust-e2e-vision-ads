import torch

# Configure accelerator/device for PyTorch and Lightning
if torch.cuda.is_available():
    ACCELERATOR = 'gpu'
    DEVICE = 0
    DEFAULT_DEVICE = f'cuda:{DEVICE}'
elif torch.backends.mps.is_available(): # Apple Silicon GPU
    ACCELERATOR = 'mps'
    DEVICE = 0
    DEFAULT_DEVICE = 'mps'
else:
    ACCELERATOR = 'cpu'
    DEVICE = 0
    DEFAULT_DEVICE = 'cpu'