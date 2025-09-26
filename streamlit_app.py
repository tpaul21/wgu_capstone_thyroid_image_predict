# File: project/streamlit_app.py
from __future__ import annotations
import io
import os
import json
from typing import Optional

import streamlit as st
import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from datetime import datetime, timezone
from project.utils_hash import verify_hash

from project.config import (
    HDF5_PATH, HDF5_IMAGES_DATASET, HDF5_LABEL_COLUMN, LABEL_MAP, OUT_DIR
)
from project.data_hdf5 import _to_pil_rgb, _transforms, _read_metadata_df, make_splits_and_loaders_mm
from project.model import build_resnet18, build_multimodal_resnet
from project.features import transform_single_row, TAB_NUM, TAB_CAT
from project.gradcam import GradCAM, overlay_cam
from project.engine import evaluate, make_criterion
from project.metrics import plot_confusion_and_roc, plot_pr_and_calibration
from project.eda import (
    plot_class_distribution, plot_tirads_distribution,
    plot_size_hist, plot_missingness, plot_numeric_correlation,
    plot_anatomical_heatmap
)

# ---------------- Page / Security ----------------
st.set_page_config(page_title="Thyroid Nodule Assistant", layout="wide")
st.title("🩺 Thyroid Nodule Decision Support Dashboard")

# Basic password gate (set STREAMLIT_PASSWORD in env to enable)
_pwd_env = os.getenv("STREAMLIT_PASSWORD")
if _pwd_env:
    pw = st.sidebar.text_input("Password", type="password", help="Protected demo")
    if pw != _pwd_env:
        st.stop()

# ---------------- Caches ----------------
@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    from project.config import CSV_PATH
    with h5py.File(HDF5_PATH, "r") as f:
        df_h5 = _read_metadata_df(f)
    df_csv = pd.read_csv(CSV_PATH)

    # Normalize join keys (keep leading zeros, strip spaces)
    for k in ("annot_id", "frame_num"):
        if k in df_h5.columns: df_h5[k] = df_h5[k].astype(str).str.strip()
        if k in df_csv.columns: df_csv[k] = df_csv[k].astype(str).str.strip()

    on_keys = [k for k in ("annot_id","frame_num") if k in df_h5.columns and k in df_csv.columns] or ["annot_id"]
    df = df_h5.merge(df_csv, on=on_keys, how="left", validate="m:1")

    if HDF5_LABEL_COLUMN in df.columns:
        df["label"] = pd.Series(df[HDF5_LABEL_COLUMN]).map(LABEL_MAP)

    for c in ("age","size_x","size_y","size_z"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_tab_transformer():
    """
    Loads the fitted ColumnTransformer from OUT_DIR. If the current feature set
    differs from the expected names saved at training time (tab_feature_names.json),
    wraps it with a feature-alignment adapter so that downstream code always
    sees the expected order/dimension.
    """
    p = OUT_DIR / "tab_transformer.joblib"
    if not p.exists():
        return None

    ct = joblib.load(p)

    expected = _load_expected_feature_names()
    if not expected:
        # No saved list; just return raw transformer
        return ct

    current = _get_feature_names(ct)
    if current is None:
        # Cannot read current names; return raw and let loader handle dim checks
        return ct

    if len(current) == len(expected) and current == expected:
        # Perfect match
        return ct

    # Mismatch in order/length: wrap with alignment adapter
    return _FeatureAligner(ct, expected_names=expected, current_names=current)


@st.cache_resource(show_spinner=False)
def load_model_and_kind():
    """
    Diagnostic, non-UI loader.

    Returns a dict:

    - {"status":"ok", "model":nn.Module, "kind":"mm"|"img", "tab_dim":int,
       "ckpt_path": str | None}

    - {"status":"dim_mismatch", "ct_dim":int, "ckpt_dim":int,
       "ckpt_path":str, "ct_path":str}

    - {"status":"no_ckpt"}
    """
    ct = load_tab_transformer()
    mm_ckpt = OUT_DIR / "best_resnet18_mm.ckpt"
    img_ckpt = OUT_DIR / "best_resnet18_hdf5.ckpt"

    # Prefer multimodal if we have a transformer and a ckpt
    if ct is not None and mm_ckpt.exists():
        # Compute current transformer output dim
        dummy = pd.DataFrame([{c: np.nan for c in (TAB_NUM + TAB_CAT)}])
        try:
            ct_dim = int(ct.transform(dummy).shape[1])
        except Exception:
            # If transform fails, fall back to image-only if possible
            ct_dim = None

        # If we can compute ct_dim, compare to what the checkpoint expects
        if ct_dim is not None:
            ckpt_dim = _ckpt_expected_tabdim(mm_ckpt)
            if ckpt_dim is not None and ckpt_dim != ct_dim:
                return {
                    "status": "dim_mismatch",
                    "ct_dim": int(ct_dim),
                    "ckpt_dim": int(ckpt_dim),
                    "ckpt_path": str(mm_ckpt.resolve()),
                    "ct_path": str((OUT_DIR / "tab_transformer.joblib").resolve()),
                }
            

            # Build and load multimodal model
            m = build_multimodal_resnet(tab_dim=ct_dim)
            m.load_state_dict(torch.load(mm_ckpt, map_location="cpu"))
            m.eval()
            return {
                "status": "ok",
                "model": m,
                "kind": "mm",
                "tab_dim": int(ct_dim),
                "ckpt_path": str(mm_ckpt.resolve()),
            }

    # Fallback: image-only if present
    if img_ckpt.exists():
        m = build_resnet18()
        m.load_state_dict(torch.load(img_ckpt, map_location="cpu"))
        m.eval()
        return {
            "status": "ok",
            "model": m,
            "kind": "img",
            "tab_dim": 0,
            "ckpt_path": str(img_ckpt.resolve()),
        }

    # Nothing available
    return {"status": "no_ckpt"}


# ---- Threshold & checkpoint helpers ----
def _read_json(path):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}

@st.cache_data(show_spinner=False)
def load_default_threshold_info() -> tuple[float, str]:
    """
    Returns (threshold, source_string).
    Prefers nodule-level val-tuned threshold, then nodule best-F1, then frame best-F1, else 0.5.
    """
    jn = _read_json(OUT_DIR / "eval_nodule_summary.json")
    if "thr_val_bestF1" in jn:
        try:
            return float(jn["thr_val_bestF1"]), "nodule val-tuned (eval_nodule_summary.json)"
        except Exception:
            pass
    if "best_F1" in jn and "threshold" in jn["best_F1"]:
        try:
            return float(jn["best_F1"]["threshold"]), "nodule best-F1 (eval_nodule_summary.json)"
        except Exception:
            pass

    jf = _read_json(OUT_DIR / "eval_summary.json")
    if "best_F1" in jf and "threshold" in jf["best_F1"]:
        try:
            return float(jf["best_F1"]["threshold"]), "frame best-F1 (eval_summary.json)"
        except Exception:
            pass
    return 0.5, "default 0.5"

# Back-compat shim if other code calls it
def load_default_threshold() -> float:
    thr, _ = load_default_threshold_info()
    return thr

def get_active_checkpoint_info():
    """
    Mirrors load_model_and_kind selection without loading the model.
    Returns (kind, ckpt_path | None), where kind in {"mm","img","none"}.
    """
    ct = load_tab_transformer()
    mm_ckpt = OUT_DIR / "best_resnet18_mm.ckpt"
    img_ckpt = OUT_DIR / "best_resnet18_hdf5.ckpt"
    if ct is not None and mm_ckpt.exists():
        return "mm", mm_ckpt
    if img_ckpt.exists():
        return "img", img_ckpt
    return "none", None

def _fmt_mtime(p):
    try:
        ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).astimezone()
        return ts.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "unknown"

# ---------------- Decision Support (prescriptive) ----------------
def triage_recommendation(prob: float, threshold: float = 0.5, margin: float = 0.10) -> str:
    # Abstain band around threshold
    if abs(prob - threshold) <= margin:
        return "Needs Review — borderline probability. Recommend expert review or repeat imaging."
    if prob >= 0.80:
        return "High risk — consider FNA biopsy / urgent specialist review."
    if prob >= 0.50:
        return "Intermediate risk — consider short-interval follow-up or additional imaging."
    return "Low risk — routine follow-up per TI-RADS and clinical context."

# ---------------- Helpers ----------------
def load_image_by_index(i: int) -> Image.Image:
    with h5py.File(HDF5_PATH, "r") as f:
        arr = f[HDF5_IMAGES_DATASET][i]
    return _to_pil_rgb(arr)

def _ckpt_expected_tabdim(ckpt_path):
    """
    Peek the checkpoint to infer the tabular in_features (expected tab_dim).
    Looks for 'tab_mlp.0.weight' which is the first linear layer in the tab MLP.
    Returns int or None if it cannot be inferred.
    """
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        w = sd.get("tab_mlp.0.weight", None)
        if w is not None and hasattr(w, "shape"):
            return int(w.shape[1])  # in_features of first MLP layer
    except Exception:
        pass
    return None

    

def predict_on_pil(
    model,
    kind: str,
    pil: Image.Image,
    x_tab: Optional[np.ndarray],
    threshold: float = 0.5,
):
    """
    kind: "mm" (multimodal) or "img" (image-only)
    Returns: pred (int), prob (float), cam (np.ndarray)
    """
    tfms = _transforms(train=False)
    x = tfms(pil).unsqueeze(0)  # CPU is fine in Streamlit

    # Forward
    if kind == "mm":
        if x_tab is None:
            ct = load_tab_transformer()
            if ct is None:
                raise RuntimeError("Multimodal model selected but no tabular transformer found.")
            df1 = pd.DataFrame([{c: np.nan for c in (TAB_NUM + TAB_CAT)}])
            x_tab = ct.transform(df1)
        xt = torch.tensor(x_tab, dtype=torch.float32)
        with torch.no_grad():
            logit = model(x, xt).squeeze(1)
    else:
        with torch.no_grad():
            logit = model(x).squeeze(1)

    prob = torch.sigmoid(logit).item()
    pred = int(prob >= threshold)

    # Grad-CAM
    if kind == "mm":
        try:
            gc = GradCAM(model, target_layer="backbone.layer4", forward_fn=lambda t: model(t, xt))
        except KeyError:
            gc = GradCAM(model, target_layer="layer4", forward_fn=lambda t: model(t, xt))
    else:
        gc = GradCAM(model, target_layer="layer4")

    cam = gc(x, class_idx=pred)
    return pred, prob, cam

def infer_and_show(
    model, kind: str, pil: Image.Image, x_tab: Optional[np.ndarray],
    threshold: float, meta_row: Optional[pd.Series] = None
):
    pred, prob, cam = predict_on_pil(model, kind, pil, x_tab, threshold)
    over = overlay_cam(pil, cam) if cam is not None else None

    st.subheader("Case Inspection")
    cols = st.columns(3)
    cols[0].image(pil, caption="Original", use_container_width=True)
    with cols[1]:
        if cam is not None:
            fig, ax = plt.subplots()
            ax.imshow(cam, cmap="jet"); ax.axis("off"); ax.set_title("Grad-CAM")
            st.pyplot(fig, clear_figure=True)
    with cols[2]:
        if over is not None:
            cols[2].image(
                over,
                caption=f"Overlay | p(malig)={prob:.2f} | decision={'M' if pred else 'B'}",
                use_container_width=True,
            )

    rec = triage_recommendation(prob, threshold=threshold, margin=0.10)
    st.markdown(f"**Recommendation:** {rec}")

    if meta_row is not None:
        st.markdown("**Metadata (redacted)**")
        meta_df = pd.DataFrame(meta_row.dropna())
        # Ensure non-numeric values are strings for Arrow
        meta_df = meta_df.applymap(lambda v: v if isinstance(v, (int, float, np.number)) else str(v))
        st.dataframe(meta_df, use_container_width=True)

def _get_feature_names(ct):
    """Return a list of current feature names from a fitted ColumnTransformer (if available)."""
    try:
        names = ct.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return None

def _load_expected_feature_names():
    """Read the feature-name list saved at training time."""
    p = OUT_DIR / "tab_feature_names.json"
    if p.exists():
        try:
            import json
            return list(json.loads(p.read_text()))
        except Exception:
            pass
    return None

class _FeatureAligner:
    """
    Wraps a fitted ColumnTransformer (ct) so that transform(df) returns columns
    in the exact order of expected_names, inserting zero-columns for any missing features.
    """
    def __init__(self, ct, expected_names, current_names):
        self.ct = ct
        self.expected = list(expected_names)
        self.current = list(current_names or [])
        self._cur_idx = {name: i for i, name in enumerate(self.current)}

    def transform(self, df):
        X = self.ct.transform(df)
        # Ensure dense array for simple placement.
        try:
            import numpy as np
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X)
            n, _ = X.shape
            out = np.zeros((n, len(self.expected)), dtype=X.dtype)
            for j, name in enumerate(self.expected):
                i = self._cur_idx.get(name, None)
                if i is not None:
                    out[:, j] = X[:, i]
                # else leave zeros for missing feature
            return out
        except Exception:
            # As a last resort return original X; downstream will error if dims mismatch.
            return X

    # Optional convenience for code that calls fit/fit_transform (we won’t use it here)
    def fit(self, df, y=None):
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)


# ---------------- Sidebar Controls ----------------
df_meta = load_metadata()

st.sidebar.header("Controls")
# Slider wired through session_state so we can set it from Performance tab
if "threshold_value" not in st.session_state or "threshold_source" not in st.session_state:
    _thr, _src = load_default_threshold_info()
    st.session_state["threshold_value"] = float(_thr)
    st.session_state["threshold_source"] = _src

threshold = st.sidebar.slider(
    "Decision Threshold (malignant)",
    0.0, 1.0, float(st.session_state["threshold_value"]), 0.001, format="%.3f", key="threshold_value"
)
_def_thr, _def_src = load_default_threshold_info()
st.sidebar.caption(f"Default threshold: **{_def_thr:.3f}** ({_def_src})")

sex_vals = sorted(df_meta["sex"].dropna().astype(str).unique().tolist()) if "sex" in df_meta else []
loc_vals = sorted(df_meta["location"].dropna().astype(str).unique().tolist()) if "location" in df_meta else []
tir_vals = sorted(df_meta["ti-rads_level"].dropna().astype(str).unique().tolist()) if "ti-rads_level" in df_meta else []

sex_sel = st.sidebar.multiselect("Sex", options=sex_vals, default=sex_vals)
loc_sel = st.sidebar.multiselect("Location", options=loc_vals, default=loc_vals)
tir_sel = st.sidebar.multiselect("TI-RADS Level", options=tir_vals, default=tir_vals)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sex" in out.columns and sex_sel: out = out[out["sex"].astype(str).isin(sex_sel)]
    if "location" in out.columns and loc_sel: out = out[out["location"].astype(str).isin(loc_sel)]
    if "ti-rads_level" in out.columns and tir_sel: out = out[out["ti-rads_level"].astype(str).isin(tir_sel)]
    return out

df_filtered = apply_filters(df_meta)
if len(df_filtered) == 0:
    st.sidebar.warning("No rows match the current filters. Clear some filters.")

# Annot ID search
search_id = st.sidebar.text_input("Find by annot_id")
if search_id and "annot_id" in df_filtered.columns:
    hits = df_filtered[df_filtered["annot_id"].astype(str).str.contains(search_id, case=False, na=False)]
else:
    hits = df_filtered

st.sidebar.markdown(f"**Filtered rows:** {len(hits):,} / {len(df_meta):,}")

# Choose a case index robustly
if "h5_index" in hits.columns and len(hits) > 0:
    default_idx = int(hits["h5_index"].iloc[0])
else:
    default_idx = 0

case_index = st.sidebar.number_input(
    "Dataset Index (after filter/search)", min_value=0, value=default_idx, step=1
)
uploaded = st.sidebar.file_uploader("Or upload an image", type=["png","jpg","jpeg","dcm"])

# ---------------- Tabs ----------------
tab_overview, tab_performance, tab_case = st.tabs(["📊 Overview", "📈 Performance", "🔍 Case Explorer"])

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        if "label" in df_meta.columns:
            st.pyplot(plot_class_distribution(df_meta), clear_figure=True)
        if "ti-rads_level" in df_meta.columns:
            st.pyplot(plot_tirads_distribution(df_meta), clear_figure=True)
    with c2:
        st.pyplot(plot_size_hist(df_meta), clear_figure=True)
        st.pyplot(plot_missingness(df_meta), clear_figure=True)
    st.pyplot(plot_numeric_correlation(df_meta), clear_figure=True)

    # Optional: anatomical heatmap from masks if present
    try:
        fig_heat = plot_anatomical_heatmap(HDF5_PATH, mask_key="/mask", max_samples=2000)
        st.pyplot(fig_heat, clear_figure=True)
    except Exception:
        st.caption("Anatomical heatmap unavailable (no /mask or incompatible).")

with tab_performance:
    st.info("Evaluates the current checkpoint on the current test split (streamed from HDF5).")

    # Model & threshold info panel (inside this tab)
    kind_used, ckpt_path = get_active_checkpoint_info()
    with st.expander("Model & threshold info", expanded=True):
        st.markdown(f"**Model kind:** {'Multimodal' if kind_used=='mm' else ('Image-only' if kind_used=='img' else 'Not found')}")
        if ckpt_path is not None:
            st.markdown(f"**Checkpoint:** `{ckpt_path.name}`")
            st.markdown(f"**Last modified:** {_fmt_mtime(ckpt_path)}")
        else:
            st.markdown("**Checkpoint:** not found")
        st.markdown(
            f"**Current Case Explorer threshold:** {st.session_state.get('threshold_value', 0.5):.3f}  \n"
            f"**Source:** {st.session_state.get('threshold_source', 'unknown')}"
        )

    col_run, col_apply = st.columns([1,1])
    
    if col_run.button("Compute / Refresh Performance"):
        st.cache_data.clear()
        st.cache_resource.clear()   # <— add this so the transformer/model reload
        st.rerun()


    res = load_model_and_kind()
    if res.get("status") == "ok":
        model, kind, _ = res["model"], res["kind"], res["tab_dim"]
    
    elif res.get("status") == "dim_mismatch":
        st.error(
            "Tabular feature dimension mismatch: transformer outputs "
            f"{res['ct_dim']} features, but checkpoint expects {res['ckpt_dim']}."
        )
        st.caption(
            "This usually means the saved transformer does not match the checkpoint. "
            "Ensure OUT_DIR contains the matching `tab_transformer.joblib` that was "
            "saved at the same time as the checkpoint."
        )
        st.write(f"**Checkpoint:** `{os.path.basename(res['ckpt_path'])}`")
        st.write(f"**Transformer:** `{os.path.basename(res['ct_path'])}`")
    
        c1, c2 = st.columns(2)
        if c1.button("Reset caches and rerun"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        if c2.button("Fallback to image-only for this session"):
            img_ckpt = OUT_DIR / "best_resnet18_hdf5.ckpt"
            if img_ckpt.exists():
                m = build_resnet18()
                m.load_state_dict(torch.load(img_ckpt, map_location="cpu"))
                m.eval()
                model, kind, _ = m, "img", 0
                st.success("Loaded image-only model for this session.")
            else:
                st.warning("Image-only checkpoint not found.")
                st.stop()
    
        # ⬇️ Do not proceed unless a model was set above
        if "model" not in locals():
            st.stop()
    
    else:
        st.warning("No trained checkpoints found. Train a model first: `python -m project.train_hdf5_mm`.")
        st.stop()


    
    train_dl, val_dl, test_dl, _, _, _ = make_splits_and_loaders_mm(batch_size=16)

    # Build pos_weight from train labels
    ys = []
    for batch in train_dl:
        y = batch[-1]
        ys.append(y.numpy())
    y_concat = np.concatenate(ys) if ys else np.array([0,1])
    pos, neg = int(y_concat.sum()), int((1 - y_concat).sum())
    criterion = make_criterion(pos, neg, device=torch.device("cpu"))

    metrics = evaluate(model, test_dl, criterion, torch.device("cpu"))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{metrics['acc']:.3f}")
    k2.metric("Precision", f"{metrics['precision']:.3f}")
    k3.metric("Recall", f"{metrics['recall']:.3f}")
    k4.metric("F1", f"{metrics['f1']:.3f}")

    # Confusion + ROC
    figs = plot_confusion_and_roc(metrics["y_true"], metrics["y_prob"], metrics["y_pred"])
    for fig in figs:
        st.pyplot(fig, clear_figure=True)

    # PR + Calibration + Brier score
    from sklearn.metrics import brier_score_loss
    for fig in plot_pr_and_calibration(metrics["y_true"], metrics["y_prob"]):
        st.pyplot(fig, clear_figure=True)
    st.caption(f"Brier score: {brier_score_loss(metrics['y_true'], metrics['y_prob']):.4f}")

    # --- Training curves from history (if available) ---
    hist_path = OUT_DIR / "train_history.jsonl"
    if hist_path.exists():
        try:
            dfh = pd.read_json(hist_path, lines=True)
            if {"epoch","train_loss","val_loss"}.issubset(dfh.columns):
                st.subheader("Training Curves")
                c1, c2 = st.columns(2)
                c1.line_chart(dfh.set_index("epoch")[["train_loss","val_loss"]])
                if {"train_acc","val_acc"}.issubset(dfh.columns):
                    c2.line_chart(dfh.set_index("epoch")[["train_acc","val_acc"]])
        except Exception as e:
            st.caption(f"Could not load train history: {e}")

    # --- Nodule-level metrics & val-tuned threshold ---
    st.subheader("Nodule-level evaluation")
    npath = OUT_DIR / "eval_nodule_summary.json"
    if npath.exists():
        try:
            nsum = _read_json(npath)

            # Display either the newer val/test schema or the older best_F1
            thr_val = None
            if "thr_val_bestF1" in nsum:
                thr_val = float(nsum["thr_val_bestF1"])
                vb = nsum.get("val_bestF1", {})
                tb = nsum.get("test_at_val_thr", {})
                st.write("**Validation (threshold tuned on validation):**")
                vcols = st.columns(5)
                vcols[0].metric("Acc", f"{vb.get('acc', float('nan')):.3f}")
                vcols[1].metric("Prec", f"{vb.get('precision', float('nan')):.3f}")
                vcols[2].metric("Rec", f"{vb.get('recall', float('nan')):.3f}")
                vcols[3].metric("F1", f"{vb.get('f1', float('nan')):.3f}")
                vcols[4].metric("AUC", f"{vb.get('auc', float('nan')):.3f}")
                st.caption(f"Val-tuned threshold: **{thr_val:.4f}**")

                st.write("**Test (applied at val-tuned threshold):**")
                tcols = st.columns(5)
                tcols[0].metric("Acc", f"{tb.get('acc', float('nan')):.3f}")
                tcols[1].metric("Prec", f"{tb.get('precision', float('nan')):.3f}")
                tcols[2].metric("Rec", f"{tb.get('recall', float('nan')):.3f}")
                tcols[3].metric("F1", f"{tb.get('f1', float('nan')):.3f}")
                tcols[4].metric("AUC", f"{tb.get('auc', float('nan')):.3f}")

            elif "best_F1" in nsum:
                best = nsum["best_F1"]
                thr_val = float(best.get("threshold", 0.5))
                cols = st.columns(5)
                cols[0].metric("Acc", f"{best.get('acc', float('nan')):.3f}")
                cols[1].metric("Prec", f"{best.get('precision', float('nan')):.3f}")
                cols[2].metric("Rec", f"{best.get('recall', float('nan')):.3f}")
                cols[3].metric("F1", f"{best.get('f1', float('nan')):.3f}")
                cols[4].metric("AUC", f"{best.get('auc', float('nan')):.3f}")
                st.caption(f"Best-F1 threshold (nodule): **{thr_val:.4f}**")

            # Apply button → updates the sidebar slider immediately and records source
            if thr_val is not None and col_apply.button("Apply val-tuned nodule threshold to Case Explorer"):
                st.session_state["threshold_value"] = float(thr_val)
                st.session_state["threshold_source"] = "nodule val-tuned (eval_nodule_summary.json)" \
                    if "thr_val_bestF1" in nsum else "nodule best-F1 (eval_nodule_summary.json)"
                st.rerun()
        except Exception as e:
            st.caption(f"Nodule summary not available: {e}")
    else:
        st.caption("Run the post-hoc nodule evaluation cell to generate eval_nodule_summary.json.")

with tab_case:
    res = load_model_and_kind()
    if res.get("status") == "ok":
        model, kind, _ = res["model"], res["kind"], res["tab_dim"]

    elif res.get("status") == "dim_mismatch":
        st.error(
            "Tabular feature dimension mismatch: transformer outputs "
            f"{res['ct_dim']} features, but checkpoint expects {res['ckpt_dim']}."
        )
        st.caption(
            "This usually means the saved transformer does not match the checkpoint. "
            "Ensure OUT_DIR contains the matching `tab_transformer.joblib` that was "
            "saved at the same time as the checkpoint."
        )
        st.write(f"**Checkpoint:** `{os.path.basename(res['ckpt_path'])}`")
        st.write(f"**Transformer:** `{os.path.basename(res['ct_path'])}`")

        c1, c2 = st.columns(2)
        if c1.button("Reset caches and rerun"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        if c2.button("Fallback to image-only for this session"):
            img_ckpt = OUT_DIR / "best_resnet18_hdf5.ckpt"
            if img_ckpt.exists():
                m = build_resnet18()
                m.load_state_dict(torch.load(img_ckpt, map_location="cpu"))
                m.eval()
                model, kind, _ = m, "img", 0
                st.success("Loaded image-only model for this session.")
            else:
                st.warning("Image-only checkpoint not found.")
                st.stop()

        # ⬇️ Do not proceed unless a model was set above
        if "model" not in locals():
            st.stop()

    else:
        st.warning("No trained checkpoints found. Train a model first: `python -m project.train_hdf5_mm`.")
        st.stop()


    

    if uploaded is not None:
        # PHI-safe DICOM parsing + RGB conversion; or plain image
        suffix = (uploaded.name.split(".")[-1] or "").lower()
        if suffix == "dcm":
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(uploaded.read()))
            # Minimal PHI scrub (extend for full policy if needed)
            for tag in [(0x0010,0x0010),(0x0010,0x0020),(0x0010,0x0030),(0x0010,0x0040)]:
                if tag in ds: del ds[tag]
            arr = ds.pixel_array
            pil = _to_pil_rgb(arr)
        else:
            pil = Image.open(uploaded).convert("RGB")

        x_tab = None
        if kind == "mm":
            ct = load_tab_transformer()
            x_tab = np.zeros(
                (1, ct.transform(pd.DataFrame([{c: np.nan for c in (TAB_NUM + TAB_CAT)}])).shape[1]),
                dtype=np.float32
            )
        infer_and_show(model, kind, pil, x_tab, threshold)

    else:
        if len(hits) == 0:
            st.warning("No rows match the current filters/search.")
        else:
            allowed = hits["h5_index"].astype(int).tolist() if "h5_index" in hits.columns else [0]
            if case_index not in allowed:
                case_index = int(allowed[0])
            pil = load_image_by_index(case_index)
            meta_row = hits[hits["h5_index"] == case_index].iloc[0] if "h5_index" in hits.columns else None
            x_tab = None
            if kind == "mm" and meta_row is not None:
                ct = load_tab_transformer()
                x_tab = transform_single_row(meta_row, ct)

            safe_cols = [c for c in [
                "annot_id","age","sex","location","size_x","size_y","size_z",
                "ti-rads_composition","ti-rads_echogenicity","ti-rads_shape",
                "ti-rads_margin","ti-rads_echogenicfoci","ti-rads_level",
                "histopath_diagnosis"
            ] if (meta_row is not None and c in hits.columns)]

            infer_and_show(model, kind, pil, x_tab, threshold, meta_row[safe_cols] if meta_row is not None else None)
