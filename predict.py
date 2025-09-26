# File: project/predict.py  (run with: python -m project.predict --image path.jpg)
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from project.config import DEVICE, OUT_DIR
from project.model import build_resnet18, build_multimodal_resnet
from project.infer import predict_with_cam

def pick_checkpoint():
    mm = OUT_DIR / "best_resnet18_mm.ckpt"
    img = OUT_DIR / "best_resnet18_hdf5.ckpt"
    legacy = OUT_DIR / "best_resnet18.ckpt"
    if mm.exists(): return "mm", mm
    if img.exists(): return "img", img
    if legacy.exists(): return "img", legacy
    return "img", img  # default path even if missing (arg may override)

def _infer_tab_dim_from_ct():
    import joblib
    ct = joblib.load(OUT_DIR / "tab_transformer.joblib")
    # Prefer the exact columns ct was fit on (features.py stamps these)
    num = list(getattr(ct, "mm_num_cols", []))
    cat = list(getattr(ct, "mm_cat_cols", []))
    if not num and not cat:
        # Fallback: use saved feature names JSON if present
        feat_path = OUT_DIR / "tab_feature_names.json"
        if feat_path.exists():
            feat_names = json.loads(feat_path.read_text())
            return len(feat_names)
        raise RuntimeError("Cannot infer tab_dim: transformer lacks mm_* cols and no feature_names.json found.")
    df_dummy = pd.DataFrame([{**{c: np.nan for c in num}, **{c: np.nan for c in cat}}])
    return ct.transform(df_dummy).shape[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image (.png/.jpg/.jpeg/.dcm)")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path (defaults to autodetect)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-show", action="store_true", help="Do not display CAM figure")
    args = parser.parse_args()

    kind, auto_ckpt = pick_checkpoint()
    ckpt_path = Path(args.ckpt) if args.ckpt else auto_ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model according to checkpoint type
    if kind == "mm":
        tab_dim = _infer_tab_dim_from_ct()
        model = build_multimodal_resnet(tab_dim=tab_dim).to(DEVICE)
    else:
        model = build_resnet18().to(DEVICE)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Note: infer.predict_with_cam auto-detects MM and generates a zero/NaN-imputed vector if possible
    pred, prob, cam, overlay = predict_with_cam(model, args.image, device=DEVICE)
    print(f"Prediction: {pred} (0=benign,1=malignant)  prob(malignant)={prob:.4f}")

if __name__ == "__main__":
    main()
