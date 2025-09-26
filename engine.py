# File: project/engine.py  (PATCHED)
from typing import Dict, Tuple
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def _forward(model, batch, device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Supports both image-only and multimodal batches
    if len(batch) == 3:
        x_img, x_tab, y = batch
        x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)
        logits = model(x_img, x_tab).squeeze(1)
    else:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
    return logits, y

def make_criterion(pos_count: int, neg_count: int, device) -> nn.Module:
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

@torch.no_grad()
def evaluate(model, dl, criterion, device) -> Dict:
    model.eval()
    losses, y_true, y_prob = [], [], []
    for batch in dl:
        logits, y = _forward(model, batch, device)
        loss = criterion(logits, y)
        losses.append(loss.item())
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float((y_pred == y_true).mean())
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if (y_true.min()!=y_true.max()) else float("nan")
    return {"loss": float(np.mean(losses)), "acc": acc, "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "auc": float(auc),
            "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

def train(model, train_dl, val_dl, device, epochs, lr, out_path: str):
    # Derive class weights from the actual labels seen in train_dl
    all_labels = []
    for batch in train_dl:
        y = batch[-1]  # last element is always labels
        all_labels.append(y.numpy() if hasattr(y, "numpy") else y.detach().cpu().numpy())
    y_concat = np.concatenate(all_labels) if all_labels else np.array([0,1])
    pos, neg = int(y_concat.sum()), int((1 - y_concat).sum())
    criterion = make_criterion(pos, neg, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # history file next to checkpoint
    hist_path = Path(out_path).with_name("train_history.jsonl")
    if hist_path.exists():
        try: hist_path.unlink()
        except Exception: pass

    best_f1 = -1.0
    for epoch in range(1, epochs+1):
        model.train()
        running_losses = []
        ep_true, ep_prob = [], []  # for train accuracy estimate

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            logits, y = _forward(model, batch, device)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_losses.append(loss.item())
            with torch.no_grad():
                ep_true.append(y.detach().cpu().numpy())
                ep_prob.append(torch.sigmoid(logits).detach().cpu().numpy())

            pbar.set_postfix(loss=float(np.mean(running_losses)))

        scheduler.step()

        # train epoch metrics
        tr_loss = float(np.mean(running_losses)) if running_losses else float("nan")
        tr_true = np.concatenate(ep_true) if ep_true else np.array([])
        tr_prob = np.concatenate(ep_prob) if ep_prob else np.array([])
        if tr_true.size > 0:
            tr_pred = (tr_prob >= 0.5).astype(int)
            tr_acc = float((tr_pred == tr_true).mean())
        else:
            tr_acc = float("nan")

        # val metrics
        val_metrics = evaluate(model, val_dl, criterion, device)
        print(f"Val: loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} "
              f"prec={val_metrics['precision']:.3f} rec={val_metrics['recall']:.3f} "
              f"f1={val_metrics['f1']:.3f} auc={val_metrics['auc']:.3f}")

        # save best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), out_path)
            print("Saved new best:", out_path)

        # append epoch history (JSONL)
        row = {
            "epoch": int(epoch),
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
        }
        try:
            with open(hist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass

    return criterion
