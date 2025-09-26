# File: project/eda.py
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

def _finish(fig, ax, tight=True):
    if tight:
        fig.tight_layout()
    return fig

def plot_class_distribution(df: pd.DataFrame, label_col: str = "label"):
    counts = df[label_col].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar([str(k) for k in counts.index], counts.values)
    ax.set_title("Class Distribution (0=benign, 1=malignant)")
    ax.set_xlabel("Class"); ax.set_ylabel("Count")
    return _finish(fig, ax)

def plot_tirads_distribution(df: pd.DataFrame, col: str = "ti-rads_level"):
    if col not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Column '{col}' not found", ha="center", va="center")
        ax.axis("off")
        return fig
    s = df[col].astype(str).dropna()
    # try numeric sort if all values look numeric
    if np.all([v.isdigit() for v in s.unique()]):
        order = sorted(s.unique(), key=lambda x: int(x))
    else:
        order = sorted(s.unique())
    counts = s.value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots()
    ax.bar(range(len(order)), counts.values)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_title(f"TI-RADS: {col}")
    ax.set_xlabel("Level"); ax.set_ylabel("Count")
    return _finish(fig, ax)

def plot_size_hist(df: pd.DataFrame, cols=("size_x","size_y","size_z")):
    fig, ax = plt.subplots()
    plotted = False
    for c in cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().values
            if len(vals) > 0:
                ax.hist(vals, bins=30, alpha=0.5, label=c)
                plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No size columns found", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.legend(); ax.set_title("Lesion Size Distributions"); ax.set_xlabel("mm"); ax.set_ylabel("Count")
    return _finish(fig, ax)

def plot_missingness(df: pd.DataFrame, top_k: int = 30):
    miss = df.isna().mean().sort_values(ascending=False).head(top_k)
    fig, ax = plt.subplots()
    ax.barh(miss.index[::-1], miss.values[::-1])
    ax.set_title("Missingness (fraction)"); ax.set_xlabel("Fraction missing"); ax.set_ylabel("Column")
    return _finish(fig, ax)

def plot_numeric_correlation(df: pd.DataFrame, include: Sequence[str] = ("age","size_x","size_y","size_z")):
    cols = [c for c in include if c in df.columns]
    if not cols:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No numeric columns to correlate", ha="center", va="center"); ax.axis("off")
        return fig
    cmat = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots()
    im = ax.imshow(cmat, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_title("Numeric Correlation Heatmap")
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cmat.values[i,j]:.2f}", ha="center", va="center")
    return _finish(fig, ax)

def plot_anatomical_heatmap(h5_path, mask_key="/mask", max_samples: int = 2000):
    """Aggregates binary masks to show frequency heatmap of lesion locations."""
    fig, ax = plt.subplots()
    try:
        with h5py.File(h5_path, "r") as f:
            if mask_key not in f:
                ax.text(0.5, 0.5, f"No mask dataset at '{mask_key}'", ha="center", va="center")
                ax.axis("off"); return fig
            m = f[mask_key]
            n = min(max_samples, m.shape[0])
            idx = np.linspace(0, m.shape[0]-1, n).astype(int)
            H, W = m.shape[1], m.shape[2]
            acc = np.zeros((H, W), dtype=np.float32)
            for i in idx:
                mi = m[i]
                if mi.ndim == 2:
                    acc += (mi > 0).astype(np.float32)
                else:
                    acc += (mi.squeeze() > 0).astype(np.float32)
            acc /= acc.max() + 1e-6
            im = ax.imshow(acc, cmap="hot", interpolation="bilinear")
            fig.colorbar(im, ax=ax)
            ax.set_title("Anatomical Heatmap (mask frequency)")
            ax.axis("off")
            return _finish(fig, ax)
    except Exception as e:
        ax.text(0.5, 0.5, f"Heatmap error: {e}", ha="center", va="center")
        ax.axis("off")
        return fig
