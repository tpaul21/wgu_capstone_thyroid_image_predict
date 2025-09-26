# File: project/data_hdf5.py
from __future__ import annotations
from typing import Sequence, Optional
import platform
import numpy as np
import pandas as pd
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from project.config import (
    HDF5_PATH, HDF5_IMAGES_DATASET, HDF5_METADATA_PATH,
    HDF5_LABEL_COLUMN, LABEL_MAP, IMG_SIZE, CSV_PATH,
    NUM_WORKERS, PIN_MEMORY
)


from project.features import fit_transform_all, transform_all

# ----------------- image helpers -----------------
def _percentile_normalize(arr: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    p1, p99 = np.percentile(a, [lo, hi])
    a = np.clip(a, p1, p99)
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return (a * 255.0).astype(np.uint8)

def _to_pil_rgb(img_np: np.ndarray) -> Image.Image:
    arr = np.asarray(img_np)
    if arr.ndim == 2:
        return Image.fromarray(_percentile_normalize(arr), mode="L").convert("RGB")
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return Image.fromarray(_percentile_normalize(arr[..., 0]), mode="L").convert("RGB")
        if arr.shape[-1] == 3 and arr.dtype == np.uint8:
            return Image.fromarray(arr[..., :3]).convert("RGB")
        gray = _percentile_normalize(arr.mean(axis=-1))
        return Image.fromarray(gray, mode="L").convert("RGB")
    raise ValueError(f"Unexpected image shape {arr.shape}")

def _transforms(train: bool):
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

# ----------------- metadata helpers -----------------
def _read_metadata_df(f: h5py.File) -> pd.DataFrame:
    """
    If HDF5_METADATA_PATH exists, read it (compound or group).
    Otherwise, build minimal index from /annot_id and /frame_num.
    Always add h5_index aligned with /image row order.
    """
    if HDF5_METADATA_PATH and HDF5_METADATA_PATH in f:
        node = f[HDF5_METADATA_PATH]
        if isinstance(node, h5py.Dataset) and node.dtype.fields:
            df = pd.DataFrame({k: node[k][...] for k in node.dtype.fields.keys()})
        elif isinstance(node, h5py.Group):
            cols = {k: v[...] for k, v in node.items() if isinstance(v, h5py.Dataset)}
            df = pd.DataFrame(cols)
        else:
            raise ValueError(f"Unsupported metadata at {HDF5_METADATA_PATH}")
    else:
        cols = {}
        if "/annot_id" in f:  cols["annot_id"]  = f["/annot_id"][...]
        if "/frame_num" in f: cols["frame_num"] = f["/frame_num"][...]
        df = pd.DataFrame(cols) if cols else pd.DataFrame()

    # Decode bytes -> strings
    for c in df.columns:
        if str(df[c].dtype).startswith("|S") or df[c].dtype == object:
            df[c] = pd.Series(df[c]).apply(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)

    # Add h5_index aligned to /image
    n = f[HDF5_IMAGES_DATASET].shape[0]
    if len(df) == 0:
        df = pd.DataFrame({"h5_index": np.arange(n)})
    else:
        # Optional safety: ensure row count matches image count
        if len(df) != n:
            # If they truly should differ, remove this assert—but for your case they should match
            raise ValueError(
                f"Metadata rows ({len(df)}) do not match image rows ({n}). "
                "Check the merge keys or HDF5_METADATA_PATH."
            )
        df = df.copy()
        df["h5_index"] = np.arange(len(df))
    return df

# ----------------- dataset -----------------
# --- Drop-in replacement for H5ThyroidDatasetMM (project/data_hdf5.py) ---
class H5ThyroidDatasetMM(Dataset):
    """
    Streams images from HDF5 and tabular vectors using separate indices.

    - row_indices: indices into the CURRENT (possibly thinned) DataFrame used to build X_tab_all & labels
    - h5_indices : indices into the original HDF5 /image dataset (global row ids)

    Returns a tuple: (x_img: Tensor[C,H,W], x_tab: Tensor[tab_dim], y: Tensor[]).
    """

    def __init__(
        self,
        row_indices: Sequence[int],
        h5_indices: Sequence[int],
        labels_row_order: np.ndarray,
        X_tab_all: np.ndarray,
        train: bool,
    ):
        assert len(row_indices) == len(h5_indices), "row_indices and h5_indices must align"
        self._row_idx = np.asarray(row_indices, dtype=np.int64)
        self._h5_idx  = np.asarray(h5_indices, dtype=np.int64)

        # Aligned to *row order* (same space as _row_idx)
        self.labels = np.asarray(labels_row_order, dtype=np.float32)
        self.X_tab  = np.asarray(X_tab_all, dtype=np.float32)  # shape [n_rows, tab_dim]

        self.train = train
        self.tfms = _transforms(train)

        # HDF5 handles (opened lazily & per-process)
        self._file: Optional[h5py.File] = None
        self._images_ds = None

        # Light sanity checks
        if self._row_idx.max(initial=-1) >= len(self.X_tab):
            raise IndexError("row_indices contain values out of bounds for X_tab_all.")
        if self._row_idx.max(initial=-1) >= len(self.labels):
            raise IndexError("row_indices contain values out of bounds for labels.")

    # -------- Public read-only accessors for sanity checks / reporting --------
    @property
    def h5_indices(self) -> np.ndarray:
        """Global HDF5 indices used to fetch images (/image rows)."""
        return self._h5_idx

    @property
    def row_indices(self) -> np.ndarray:
        """Row indices into the current (thinned) DataFrame / X_tab_all / labels."""
        return self._row_idx

    # -------- HDF5 lifecycle helpers (safe across workers) --------
    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(HDF5_PATH, "r")
            self._images_ds = self._file[HDF5_IMAGES_DATASET]

    def close(self):
        """Close any open HDF5 handles (safe to call multiple times)."""
        try:
            if self._images_ds is not None:
                self._images_ds = None
            if self._file is not None:
                try:
                    self._file.close()
                finally:
                    self._file = None
        except Exception:
            # Avoid raising during GC
            self._file = None
            self._images_ds = None

    def __del__(self):
        # Best-effort cleanup
        self.close()

    # Make sure workers don't inherit open handles; they will reopen lazily
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_images_ds"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Handles will be reopened on first __getitem__
        self._file = None
        self._images_ds = None

    # -------- PyTorch Dataset protocol --------
    def __len__(self) -> int:
        return len(self._row_idx)

    def __getitem__(self, i: int):
        self._ensure_open()

        r = int(self._row_idx[i])  # index into X_tab_all / labels
        h = int(self._h5_idx[i])   # index into HDF5 images

        # Image → PIL → tensor
        img_np = self._images_ds[h]
        pil = _to_pil_rgb(img_np)
        x_img = self.tfms(pil)

        # Tabular + label from row space
        x_tab = torch.from_numpy(self.X_tab[r])
        y     = torch.tensor(self.labels[r], dtype=torch.float32)

        return x_img, x_tab, y




# ----------------- loaders -----------------
def make_splits_and_loaders_mm(
    batch_size: int,
    seed: int = 42,
    subset_groups_frac: Optional[float] = None,
    subset_groups_max: Optional[int] = None,
    frame_stride: int = 1,
    max_frames_per_group: Optional[int] = None,
    samples_per_epoch: Optional[int] = None,
    # NEW: split fractions + malignant minimums (nodule-level)
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    min_mal_val: int = 4,
    min_mal_test: int = 4,
):
    """
    Build train/val/test loaders with optional subsetting and capped-epoch sampling.

    - subset_groups_frac: keep a random fraction of annot_id groups (e.g., 0.2)
    - subset_groups_max:  keep at most N annot_id groups
    - frame_stride:       within each annot_id, take every k-th frame (e.g., 5)
    - max_frames_per_group: cap frames per annot_id (e.g., 10)
    - samples_per_epoch:  if set, train_dl uses RandomSampler(replacement=True, num_samples=...)
    - train/val/test fractions sum to 1.0 (nodule-level when annot_id is available)
    - min_mal_val/test ensure at least this many malignant nodules in val/test (if available)
    """
    import platform
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, RandomSampler

    # 1) Read HDF5 minimal metadata and merge with CSV
    with h5py.File(HDF5_PATH, "r") as f:
        df_h5 = _read_metadata_df(f)  # includes original h5_index (0..N-1), aligned to /image
    df_h5 = df_h5.copy()  # keep name stable

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {CSV_PATH}. It must include '{HDF5_LABEL_COLUMN}' "
            "and join keys like 'annot_id' (and optionally 'frame_num')."
        )
    df_csv = pd.read_csv(CSV_PATH)

    # Normalize join keys
    for k in ("annot_id", "frame_num"):
        if k in df_h5.columns:  df_h5[k]  = df_h5[k].astype(str).str.strip()
        if k in df_csv.columns: df_csv[k] = df_csv[k].astype(str).str.strip()

    on_keys = [k for k in ("annot_id","frame_num") if k in df_h5.columns and k in df_csv.columns] or ["annot_id"]
    df = df_h5.merge(df_csv, on=on_keys, how="left", validate="m:1").copy()

    # Preserve original HDF5 row ids (for image streaming)
    if "h5_index" not in df.columns:
        df["h5_index"] = np.arange(len(df), dtype=np.int64)

    # 2) Optional: dataset-level thinning by group and per-group frames
    if "annot_id" in df.columns:
        rng = np.random.RandomState(seed)
        groups_all = df["annot_id"].astype(str).unique()
        rng.shuffle(groups_all)

        keep_groups = list(groups_all)
        if subset_groups_frac is not None:
            k = max(1, int(len(groups_all) * float(subset_groups_frac)))
            keep_groups = keep_groups[:k]
        if subset_groups_max is not None:
            keep_groups = keep_groups[: int(subset_groups_max)]

        if (subset_groups_frac is not None) or (subset_groups_max is not None):
            df = df[df["annot_id"].astype(str).isin(set(keep_groups))].copy()

        # Per-group frame thinning
        if frame_stride > 1 or (max_frames_per_group is not None):
            def _thin(g: pd.DataFrame) -> pd.DataFrame:
                if "frame_num" in g.columns:
                    order = pd.to_numeric(g["frame_num"], errors="coerce").values
                else:
                    order = np.arange(len(g))
                g = g.iloc[np.argsort(np.where(np.isnan(order), np.inf, order))]
                g = g.iloc[::max(1, int(frame_stride))]
                if max_frames_per_group is not None:
                    g = g.head(int(max_frames_per_group))
                return g

            gb = df.groupby("annot_id", group_keys=False)
            # pandas >= 2.2 supports include_groups; keep a fallback for older versions
            try:
                df = gb.apply(_thin, include_groups=False).reset_index(drop=True)
            except TypeError:
                df = gb.apply(_thin).reset_index(drop=True)

    # 3) Labels -> {0,1} (aligned to CURRENT df row order)
    if HDF5_LABEL_COLUMN not in df.columns:
        raise KeyError(f"Label column '{HDF5_LABEL_COLUMN}' not found after merge.")
    labels_series = pd.Series(df[HDF5_LABEL_COLUMN]).map(LABEL_MAP)
    if labels_series.isna().any():
        bad = df.loc[labels_series.isna(), HDF5_LABEL_COLUMN].value_counts().to_dict()
        raise ValueError(f"Unmapped label values in {HDF5_LABEL_COLUMN}: {bad}")
    labels = labels_series.astype(np.int64).values

    # Keep both row and h5 indices
    df = df.reset_index(drop=True)
    df["row_idx"] = np.arange(len(df), dtype=np.int64)  # for X_all / labels

    # 4) Split by annot_id groups when present (preferred), else frame-level fallback
    if "annot_id" in df.columns:
        rng = np.random.RandomState(seed)
        # one label per group (first occurrence)
        gtab = (df.groupby("annot_id")[HDF5_LABEL_COLUMN]
                  .first().map(LABEL_MAP).astype(int).rename("y")).reset_index()
        groups_all = gtab["annot_id"].astype(str).values
        y_all      = gtab["y"].values

        # target split sizes (by NODULES)
        if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
        nG = len(groups_all)
        n_val   = max(1, int(round(nG * val_frac)))
        n_test  = max(1, int(round(nG * test_frac)))
        n_train = max(1, nG - n_val - n_test)

        # malignant vs benign pools
        mal_groups = gtab.loc[gtab["y"] == 1, "annot_id"].astype(str).values
        ben_groups = gtab.loc[gtab["y"] == 0, "annot_id"].astype(str).values
        rng.shuffle(mal_groups)
        rng.shuffle(ben_groups)

        # desired malignant per split (respect minimums; clamp if not enough total)
        tgt_val_m  = max(min_mal_val,  int(round(len(mal_groups) * val_frac)))
        tgt_test_m = max(min_mal_test, int(round(len(mal_groups) * test_frac)))
        if tgt_val_m + tgt_test_m > len(mal_groups):
            # balance as best as possible
            half = len(mal_groups) // 2
            tgt_val_m  = max(min_mal_val, half)
            tgt_test_m = len(mal_groups) - tgt_val_m

        # allocate malignant first
        val_m  = list(mal_groups[:tgt_val_m])
        test_m = list(mal_groups[tgt_val_m:tgt_val_m + tgt_test_m])
        train_m = list(mal_groups[tgt_val_m + tgt_test_m:])

        # fill with benign to reach target sizes
        remain_val  = max(0, n_val  - len(val_m))
        remain_test = max(0, n_test - len(test_m))
        val_b  = list(ben_groups[:remain_val])
        test_b = list(ben_groups[remain_val:remain_val + remain_test])
        ben_used = remain_val + remain_test
        train_b  = list(ben_groups[ben_used:])

        g_val   = set(val_m + val_b)
        g_test  = set(test_m + test_b)
        g_train = set(train_m + train_b)

        # re-balance sizes to hit exact targets (prefer moving benign)
        def _move_some(src: set, dst: set, need: int):
            if need <= 0: return
            # move benign first
            src_ben = [g for g in src if g in set(train_b) or g in set(val_b) or g in set(test_b)]
            move = src_ben[:need] if len(src_ben) >= need else list(src)[:need]
            for m in move:
                if m in src: src.remove(m)
                dst.add(m)

        # If val/test short, pull from train
        if len(g_val) < n_val:  _move_some(g_train, g_val,  n_val - len(g_val))
        if len(g_test) < n_test: _move_some(g_train, g_test, n_test - len(g_test))
        # If val/test long (shouldn't happen often), push extras back to train
        if len(g_val) > n_val:  _move_some(g_val,  g_train, len(g_val)  - n_val)
        if len(g_test) > n_test: _move_some(g_test, g_train, len(g_test) - n_test)
        # Final nudge if train is off due to rounding
        if len(g_train) != n_train:
            diff = len(g_train) - n_train
            if diff > 0:
                _move_some(g_train, g_val,  min(diff, n_val - len(g_val)))
                _move_some(g_train, g_test, min(diff, n_test - len(g_test)))
            elif diff < 0:
                _move_some(g_val,  g_train, min(len(g_val),  -diff))
                _move_some(g_test, g_train, min(len(g_test), -diff))

        # Safety: no group leakage
        assert len(g_train & g_val) == 0 and len(g_train & g_test) == 0 and len(g_val & g_test) == 0, \
            "Group leakage detected."

        # map groups to frame indices
        m_tr = df["annot_id"].astype(str).isin(g_train).values
        m_va = df["annot_id"].astype(str).isin(g_val).values
        m_te = df["annot_id"].astype(str).isin(g_test).values

        # row indices (for X_all & labels)
        tr_rows = np.nonzero(m_tr)[0]
        va_rows = np.nonzero(m_va)[0]
        te_rows = np.nonzero(m_te)[0]
        # h5 indices (for images)
        tr_h5 = df.loc[m_tr, "h5_index"].values
        va_h5 = df.loc[m_va, "h5_index"].values
        te_h5 = df.loc[m_te, "h5_index"].values

    else:
        # Fallback: frame-level stratified split on current df rows
        all_rows = np.arange(len(df))
        y = labels
        # first split off (val+test)
        vt_size = val_frac + test_frac
        train_rows, tmp_rows, y_train, y_tmp = train_test_split(
            all_rows, y, test_size=vt_size, stratify=y, random_state=seed
        )
        # now split tmp into val/test by their relative proportions
        val_ratio = val_frac / max(vt_size, 1e-8)
        val_rows, test_rows, y_val, y_test = train_test_split(
            tmp_rows, y_tmp, test_size=(1 - val_ratio), stratify=y_tmp, random_state=seed+1
        )
        tr_rows, va_rows, te_rows = train_rows, val_rows, test_rows
        tr_h5   = df.loc[tr_rows, "h5_index"].values
        va_h5   = df.loc[va_rows, "h5_index"].values
        te_h5   = df.loc[te_rows,  "h5_index"].values

    # 5) Tabular features: fit on train rows (current df), transform all current rows
    df_train = df.iloc[tr_rows]
    df_val   = df.iloc[va_rows]
    df_test  = df.iloc[te_rows]

    X_tr, X_va, X_te, ct, feat_names = fit_transform_all(df_train, df_val, df_test)
    X_all = transform_all(df, ct)  # aligned to df row order (row_idx)
    tab_dim = X_all.shape[1]

    # 6) Build datasets using BOTH row_indices and h5_indices (dual-index to avoid X_tab misalignment)
    train_ds = H5ThyroidDatasetMM(tr_rows, tr_h5, labels, X_all, train=True)
    val_ds   = H5ThyroidDatasetMM(va_rows, va_h5, labels, X_all, train=False)
    test_ds  = H5ThyroidDatasetMM(te_rows, te_h5, labels, X_all, train=False)

    # 7) Loaders (optionally capped per-epoch)
    # Prefer config knobs if present; else fall back to safe defaults
    try:
        from project.config import NUM_WORKERS as _NW, PIN_MEMORY as _PM
        num_workers = _NW
        pin_memory  = _PM
    except Exception:
        is_windows = platform.system().lower().startswith("win")
        num_workers = 0 if is_windows else 2
        pin_memory  = False if is_windows else True

    if samples_per_epoch is not None:
        sampler = RandomSampler(train_ds, replacement=True, num_samples=int(samples_per_epoch))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    val_dl  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)

    return train_dl, val_dl, test_dl, ct, feat_names, tab_dim

