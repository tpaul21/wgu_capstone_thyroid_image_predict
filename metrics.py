# File: project/metrics.py  (REPLACE)
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

def plot_confusion_and_roc(y_true, y_prob, y_pred):
    figs = []

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(2), yticks=range(2),
        xticklabels=["benign","malignant"], yticklabels=["benign","malignant"],
        ylabel='True', xlabel='Pred', title='Confusion Matrix'
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    figs.append(fig_cm)

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig_roc, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax2.plot([0,1],[0,1],'--')
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC"); ax2.legend()
        figs.append(fig_roc)
    except Exception:
        pass

    return figs

def plot_pr_and_calibration(y_true, y_prob):
    figs = []

    # Precision-Recall Curve
    try:
        p, r, _ = precision_recall_curve(y_true, y_prob)
        fig_pr, ax = plt.subplots()
        ax.plot(r, p)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall Curve")
        figs.append(fig_pr)
    except Exception:
        pass

    # Reliability / Calibration curve (10 bins)
    try:
        y_true = np.asarray(y_true).astype(float)
        y_prob = np.asarray(y_prob).astype(float)
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(y_prob, bins) - 1
        frac_pos, mean_pred = [], []
        for b in range(len(bins) - 1):
            m = (idx == b)
            if m.any():
                frac_pos.append(y_true[m].mean())
                mean_pred.append(y_prob[m].mean())
        fig_cal, ax2 = plt.subplots()
        ax2.plot([0, 1], [0, 1], '--', label="Perfect")
        ax2.plot(mean_pred, frac_pos, marker='o', label="Model")
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Observed positive rate")
        ax2.set_title("Reliability Curve")
        ax2.legend()
        figs.append(fig_cal)
    except Exception:
        pass

    return figs

