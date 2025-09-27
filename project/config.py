# File: project/config.py
from pathlib import Path
import os
import random
import numpy as np
import torch

# =========================
# Paths & discovery
# =========================
BASE_DIR   = Path(__file__).resolve().parent      # .../project
REPO_ROOT  = BASE_DIR.parent                      # repo root

# --- Outputs: prefer repo-root /outputs (where your ckpt/json live) ---
OUT_DIR = Path(os.getenv("OUT_DIR", REPO_ROOT / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Data: you placed samples in project/; fall back to project/data or repo/data ---
def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

env_h5  = os.getenv("HDF5_PATH")
env_csv = os.getenv("CSV_PATH")

HDF5_PATH = _first_existing([
    Path(env_h5) if env_h5 else None,
    BASE_DIR / "dataset_sample.hdf5",
    BASE_DIR / "data" / "dataset.hdf5",
    REPO_ROOT / "data" / "dataset.hdf5",
    BASE_DIR / "data" / "dataset_sample.hdf5",
    REPO_ROOT / "data" / "dataset_sample.hdf5",
]) or (BASE_DIR / "dataset_sample.hdf5")

CSV_PATH = _first_existing([
    Path(env_csv) if env_csv else None,
    BASE_DIR / "metadata_sample.csv",
    BASE_DIR / "data" / "metadata.csv",
    REPO_ROOT / "data" / "metadata.csv",
    BASE_DIR / "data" / "metadata_sample.csv",
    REPO_ROOT / "data" / "metadata_sample.csv",
]) or (BASE_DIR / "metadata_sample.csv")

# =========================
# Dataset schema
# =========================
HDF5_IMAGES_DATASET = "/image"
HDF5_METADATA_PATH  = None
HDF5_LABEL_COLUMN   = "histopath_diagnosis"

LABEL_MAP = {
    "benign": 0, "Benign": 0, 0: 0,
    "malignant": 1, "Malignant": 1, 1: 1,
}

# =========================
# Training defaults
# =========================
SEED = 123
IMG_SIZE = 256
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 3

# CPU on Streamlit Cloud; CUDA locally if available.
# You can force cloud mode by setting CLOUD=1 in Streamlit secrets.
DEVICE = torch.device(
    "cpu" if os.getenv("CLOUD") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")
)

# =========================
# Fast-training knobs
# =========================
SUBSET_GROUPS_FRAC   = 0.75
SUBSET_GROUPS_MAX    = None
FRAME_STRIDE         = 1
MAX_FRAMES_PER_GROUP = 16
SAMPLES_PER_EPOCH    = 2500
NUM_WORKERS          = 0
PIN_MEMORY           = False

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
