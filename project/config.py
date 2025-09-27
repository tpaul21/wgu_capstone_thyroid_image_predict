# File: project/config.py
from pathlib import Path
import os
import random
import numpy as np
import torch

# ---------------- Paths & discovery ----------------
BASE_DIR = Path(__file__).resolve().parent        # .../project
REPO_ROOT = BASE_DIR.parent                       # repo root
DATA_DIRS = [
    BASE_DIR / "data",            # preferred (project/data)
    BASE_DIR,                     # project/ (your current placement)
    REPO_ROOT / "data",           # repo-root/data (fallback)
]

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _first_existing(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None

# Candidate filenames (full + sample)
_hdf5_candidates = [
    d / "dataset.hdf5" for d in DATA_DIRS
] + [
    d / "dataset_sample.hdf5" for d in DATA_DIRS
]

_csv_candidates = [
    d / "metadata.csv" for d in DATA_DIRS
] + [
    d / "metadata_sample.csv" for d in DATA_DIRS
]

# Allow explicit overrides via environment variables
env_h5 = os.getenv("HDF5_PATH")
env_csv = os.getenv("CSV_PATH")

HDF5_PATH = Path(env_h5) if env_h5 else _first_existing(_hdf5_candidates)
CSV_PATH  = Path(env_csv) if env_csv else _first_existing(_csv_candidates)

# Final fallback so attributes always exist (app will show a clear error if missing)
if HDF5_PATH is None:
    HDF5_PATH = BASE_DIR / "dataset_sample.hdf5"   # matches your current layout
if CSV_PATH is None:
    CSV_PATH = BASE_DIR / "metadata_sample.csv"

# ---------------- Dataset schema ----------------
HDF5_IMAGES_DATASET = "/image"
HDF5_METADATA_PATH  = None
HDF5_LABEL_COLUMN   = "histopath_diagnosis"

# Map labels (make this match your CSV)
LABEL_MAP = {
    "benign": 0, "Benign": 0, 0: 0,
    "malignant": 1, "Malignant": 1, 1: 1
}

# ---------------- Training defaults ----------------
SEED = 123
IMG_SIZE = 256
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 3

# CPU on Streamlit Cloud; CUDA locally if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Fast-training knobs ----------------
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
