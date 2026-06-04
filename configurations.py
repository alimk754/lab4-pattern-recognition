import random
import numpy as np
import torch
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_ROOT = Path(r"C:\Users\PC\Downloads\UrbanSound8K\UrbanSound8K")
AUDIO_DIR    = DATASET_ROOT / "audio"
META_CSV     = DATASET_ROOT / "metadata" / "UrbanSound8K.csv"

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing",
    "dog_bark", "drilling", "engine_idling",
    "gun_shot", "jackhammer", "siren", "street_music",
]

# Audio / feature constants
SR         = 22050
N_MFCC     = 40
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048

# Training constants
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
SEED       = 42

# ==============================================================================
# REPRODUCIBILITY & DEVICE
# ==============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")