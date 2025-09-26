# File: project/config.py  (drop-in patch for paths)
from pathlib import Path
import random, numpy as np, torch


# Base this on the file location, not the working directory
BASE_DIR = Path(__file__).resolve().parent            # .../project
DATA_DIR = BASE_DIR / "data"                          # .../project/data
OUT_DIR  = BASE_DIR / "outputs"                       # .../project/outputs

HDF5_PATH = DATA_DIR / "dataset.hdf5"
CSV_PATH  = DATA_DIR / "metadata.csv"

# Internal HDF5 keys (from your dump)
HDF5_IMAGES_DATASET = "/image"
HDF5_METADATA_PATH  = None
HDF5_LABEL_COLUMN   = "histopath_diagnosis"

# Training defaults
SEED = 123
IMG_SIZE = 256
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {"benign": 0, "Benign": 0, 0: 0, "malignant": 1, "Malignant": 1, 1: 1}

# ---- Fast-training knobs (optional) ----
SUBSET_GROUPS_FRAC = 0.75   # e.g., 0.2 keeps ~20% of annot_id groups
SUBSET_GROUPS_MAX  = None   # e.g., 300 keeps first 300 annot_id groups (after shuffle)
FRAME_STRIDE       = 1      # e.g., 5 keeps every 5th frame within each annot_id
MAX_FRAMES_PER_GROUP = 16 # e.g., 10 caps frames per annot_id to 10
SAMPLES_PER_EPOCH  = 2500   # e.g., 30000 => train sampler draws 30k samples/epoch (with replacement)
NUM_WORKERS        = 0      # try 2 on Windows; set 0 if you hit HDF5 multiprocessing issues
PIN_MEMORY         = False  # True on Linux CUDA; keep False on Windows if you see issues


def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Cloud demo mode (Streamlit Cloud) ---
import os
if os.getenv("CLOUD") == "1":
    # Use the small sample that lives under project/data
    HDF5_PATH = BASE_DIR / "data" / "dataset_sample.hdf5"
    CSV_PATH  = BASE_DIR / "data" / "metadata_sample.csv"
    OUT_DIR   = BASE_DIR / "outputs"
    DEVICE    = torch.device("cpu")  # Streamlit Cloud is CPU-only

    # If your HDF5 sample uses these keys (from our script below):
    try:
        HDF5_IMAGES_DATASET
    except NameError:
        pass
    HDF5_IMAGES_DATASET = "/image"
    # No compound metadata table inside HDF5; we merge with CSV
    HDF5_METADATA_PATH = ""  # empty means "build minimal index from annot_id/frame_num"

    
