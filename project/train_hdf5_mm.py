# File: project/train_hdf5_mm.py
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CLI runs
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from project.utils_hash import write_hash_for, verify_hash  


from project.config import (
    OUT_DIR, DEVICE, BATCH_SIZE, LR, EPOCHS, set_seed,    
    SUBSET_GROUPS_FRAC, SUBSET_GROUPS_MAX, FRAME_STRIDE,
    MAX_FRAMES_PER_GROUP, SAMPLES_PER_EPOCH
)

from project.data_hdf5 import make_splits_and_loaders_mm
from project.model import build_multimodal_resnet
from project.engine import train, evaluate


def _save_confusion_and_roc(y_true, y_prob, y_pred, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(cm, interpolation="nearest")
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(
        xticks=range(2), yticks=range(2),
        xticklabels=["benign", "malignant"], yticklabels=["benign", "malignant"],
        ylabel="True", xlabel="Pred", title="Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig1.tight_layout()
    fig1.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig1)

    # ROC
    if np.min(y_true) != np.max(y_true):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    else:
        fpr, tpr, auc = [0, 1], [0, 1], float("nan")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax2.plot([0, 1], [0, 1], "--")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC"); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "roc_curve.png", dpi=160)
    plt.close(fig2)


def main():
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # NEW: pass fast-training knobs through to the loader
    train_dl, val_dl, test_dl, ct, feat_names, tab_dim = make_splits_and_loaders_mm(
        batch_size=BATCH_SIZE,
        subset_groups_frac=SUBSET_GROUPS_FRAC,
        subset_groups_max=SUBSET_GROUPS_MAX,
        frame_stride=FRAME_STRIDE,
        max_frames_per_group=MAX_FRAMES_PER_GROUP,
        samples_per_epoch=SAMPLES_PER_EPOCH,
    )

    # Handy split sizes (sanity check when subsetting)
    print(
        f"Split sizes — train:{len(train_dl.dataset)}  "
        f"val:{len(val_dl.dataset)}  test:{len(test_dl.dataset)}  "
        f"tab_dim:{tab_dim}"
    )

    # Persist tabular transformer + feature names
    joblib.dump(ct, OUT_DIR / "tab_transformer.joblib")
    with open(OUT_DIR / "tab_feature_names.json", "w") as f:
        json.dump(feat_names, f, indent=2)

    # Build multimodal model
    model = build_multimodal_resnet(tab_dim=tab_dim).to(DEVICE)
    ckpt = str(OUT_DIR / "best_resnet18_mm.ckpt")

    # Train & save best
    criterion = train(model, train_dl, val_dl, DEVICE, EPOCHS, LR, ckpt)

    # --- NEW: write SHA-256 for the best checkpoint we just saved ---
    if ckpt_path.exists():
        write_hash_for(ckpt_path)
        print(f"Wrote SHA-256 for {ckpt_path.name}")
    else:
        print(f"WARNING: expected checkpoint not found: {ckpt_path}")

    # --- NEW: verify SHA-256 before loading the checkpoint ---
    if not verify_hash(ckpt_path):
        raise RuntimeError(
            f"Checkpoint hash mismatch for {ckpt_path}. "
            "Delete the file and retrain to regenerate a valid checkpoint."
        )
    
    # Load best and evaluate on held-out test
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    metrics = evaluate(model, test_dl, criterion, DEVICE)

    # Scalar-only print for clean logs
    printable = {
        k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else "...")
        for k, v in metrics.items()
    }
    print("Test:", printable)

    # Persist scalar metrics to JSON (omit arrays)
    with open(OUT_DIR / "metrics_mm.json", "w") as f:
        json.dump(
            {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                for k, v in metrics.items()
                if k not in ("y_true", "y_pred", "y_prob")
            },
            f,
            indent=2,
        )

    # Save CM + ROC plots (non-blocking)
    _save_confusion_and_roc(metrics["y_true"], metrics["y_prob"], metrics["y_pred"], OUT_DIR)


if __name__ == "__main__":
    main()
