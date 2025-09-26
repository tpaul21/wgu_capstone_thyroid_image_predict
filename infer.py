# File: project/infer.py  (REPLACE)
from pathlib import Path
import torch
from PIL import Image
from project.config import DEVICE, OUT_DIR
from project.data_hdf5 import _transforms, _to_pil_rgb
from project.gradcam import GradCAM, overlay_cam

def load_for_infer(path: str):
    p = Path(path)
    if p.suffix.lower() == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(p))
        pil = _to_pil_rgb(ds.pixel_array)
    else:
        pil = Image.open(p).convert("RGB")
    x = _transforms(train=False)(pil).unsqueeze(0).to(DEVICE)
    return pil, x

@torch.no_grad()
def predict_with_cam(model, image_path: str, x_tab=None, target_layer="layer4", device=DEVICE):
    pil, x = load_for_infer(image_path)
    model.eval()

    if x_tab is None:
        # Try to detect multimodal and create a zeros feature vector using saved transformer
        try:
            import joblib, pandas as pd, numpy as np
            ct_path = OUT_DIR / "tab_transformer.joblib"
            if ct_path.exists():
                ct = joblib.load(ct_path)
                # Create a single-row NaN frame with expected columns; transform -> zeros via imputer
                from project.features import TAB_NUM, TAB_CAT
                data = {c: [float("nan")] for c in (TAB_NUM + TAB_CAT)}
                xt_np = ct.transform(pd.DataFrame(data))
                x_tab = torch.tensor(xt_np, dtype=torch.float32, device=device)
                target_layer = "backbone.layer4"
        except Exception:
            x_tab = None  # fallback to image-only

    # Forward
    if x_tab is not None:
        logit = model(x, x_tab).squeeze(1)
        gc = GradCAM(model, target_layer=target_layer, forward_fn=lambda t: model(t, x_tab))
    else:
        logit = model(x).squeeze(1)
        gc = GradCAM(model, target_layer=target_layer)

    prob = torch.sigmoid(logit).item()
    pred = int(prob >= 0.5)
    cam = gc(x, class_idx=pred)
    overlay = overlay_cam(pil, cam)

    return pred, prob, cam, overlay
