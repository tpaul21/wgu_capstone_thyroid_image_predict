# File: project/features.py
from __future__ import annotations
from typing import List, Tuple, Sequence, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------
# Whitelisted clinical/descriptive features ONLY.
# (Identifiers, labels, and post-biopsy variables are intentionally excluded.)
# ---------------------------------------------------------------------
TAB_NUM: List[str] = [
    "age",
    "size_x", "size_y", "size_z",  # lesion dimensions
]

TAB_CAT: List[str] = [
    "sex",
    "location",
    "ti-rads_composition",
    "ti-rads_echogenicity",
    "ti-rads_shape",
    "ti-rads_margin",
    "ti-rads_echogenicfoci",
    "ti-rads_level",
]

# Columns we must never include as features
_EXCLUDE_ALWAYS = {
    "histopath_diagnosis",  # the label
    "label",                # any derived label column
}

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _safe_onehot(*, handle_unknown: str = "ignore"):
    """Create a dense OneHotEncoder that works across sklearn versions."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)

def _select_available_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Intersect allowed columns with what's actually in df.
    Exclude any columns on the _EXCLUDE_ALWAYS list.
    """
    cols = set(df.columns)
    num = [c for c in TAB_NUM if c in cols and c not in _EXCLUDE_ALWAYS]
    cat = [c for c in TAB_CAT if c in cols and c not in _EXCLUDE_ALWAYS]

    if len(num) == 0 and len(cat) == 0:
        raise ValueError(
            "No allowed clinical features found in DataFrame. "
            f"Expected any of NUM={TAB_NUM} or CAT={TAB_CAT}; "
            f"df has: {list(df.columns)}"
        )
    return num, cat

def _build_ct(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer with:
      - numeric: impute median -> standardize
      - categorical: impute 'missing' -> one-hot (dense)
    """
    transformers = []
    if num_cols:
        num_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe",    _safe_onehot()),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        raise ValueError("No transformers configured (no numeric or categorical columns).")

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=None,
        verbose_feature_names_out=False,  # cleaner names
    )
    # Keep track of the columns actually used (for robust transforms later)
    ct.mm_num_cols = list(num_cols)
    ct.mm_cat_cols = list(cat_cols)
    return ct

def _get_feature_names(ct: ColumnTransformer) -> List[str]:
    """Return expanded feature names after fit."""
    names: List[str] = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if name == "num":
            names.extend(list(cols))
        elif name == "cat":
            ohe = trans.named_steps.get("ohe")
            if hasattr(ohe, "get_feature_names_out"):
                names.extend(list(ohe.get_feature_names_out(cols)))
            else:
                names.extend(list(ohe.get_feature_names(cols)))
        else:
            names.extend(list(cols))
    return names

def _coerce_dtypes(df: pd.DataFrame, num_cols: Sequence[str], cat_cols: Sequence[str]) -> pd.DataFrame:
    """Ensure numeric are numeric, categorical are strings; do not mutate original df."""
    dfc = df.copy()
    for c in num_cols:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
    for c in cat_cols:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype(str)
    return dfc

def _extract_mm_cols_from_ct(maybe_ct) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Try to recover the numeric/categorical column lists the CT was fit on.
    Works if:
      - ct has .mm_num_cols/.mm_cat_cols (ours), OR
      - ct is a wrapper with inner CT on attribute .ct that has those.
    Returns (num_cols, cat_cols) or (None, None) if unavailable.
    """
    for obj in (maybe_ct, getattr(maybe_ct, "ct", None)):
        if obj is None:
            continue
        num = getattr(obj, "mm_num_cols", None)
        cat = getattr(obj, "mm_cat_cols", None)
        if isinstance(num, (list, tuple)) and isinstance(cat, (list, tuple)):
            return list(num), list(cat)
    return None, None

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def fit_transform_all(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Fit a tabular preprocessor on TRAIN ONLY, then transform train/val/test.
    Returns X_train, X_val, X_test (float32), the fitted ColumnTransformer, and feature names.
    """
    # 1) pick available columns on TRAIN
    num_cols, cat_cols = _select_available_columns(df_train)

    # 2) build CT
    ct = _build_ct(num_cols, cat_cols)

    # 3) coerce dtypes & fit
    df_train_c = _coerce_dtypes(df_train, num_cols, cat_cols)
    X_tr = ct.fit_transform(df_train_c)
    feat_names = _get_feature_names(ct)

    # 4) transform val/test
    df_val_c = _coerce_dtypes(df_val, num_cols, cat_cols)
    df_test_c = _coerce_dtypes(df_test, num_cols, cat_cols)
    X_va = ct.transform(df_val_c)
    X_te = ct.transform(df_test_c)

    # cast to float32 for PyTorch
    X_tr = X_tr.astype(np.float32, copy=False)
    X_va = X_va.astype(np.float32, copy=False)
    X_te = X_te.astype(np.float32, copy=False)
    return X_tr, X_va, X_te, ct, feat_names

def transform_all(df: pd.DataFrame, ct) -> np.ndarray:
    """
    Transform an arbitrary DataFrame with an already-fitted ColumnTransformer or a wrapper
    exposing .transform(df). We prefer the exact numeric/categorical columns the CT was fit on.
    If those are not discoverable, we fall back to the whitelisted TAB_NUM/TAB_CAT present in df.
    Returns float32 numpy array [N, D].
    """
    # Try to recover the exact columns the CT expects
    num_cols, cat_cols = _extract_mm_cols_from_ct(ct)

    if num_cols is None or cat_cols is None:
        # Fallback: use the whitelist intersected with df (safe because CT will ignore unknowns via column names)
        avail_num, avail_cat = _select_available_columns(df)
        num_cols = avail_num
        cat_cols = avail_cat

    # Build a projection DataFrame with just the columns we want; missing cols become NaN
    data = {}
    for c in num_cols:
        data[c] = pd.to_numeric(df.get(c, pd.Series([np.nan] * len(df))), errors="coerce")
    for c in cat_cols:
        s = df.get(c, pd.Series([np.nan] * len(df)))
        data[c] = s.astype(str)

    dfx = pd.DataFrame(data, index=df.index)
    X = ct.transform(dfx).astype(np.float32, copy=False)
    return X

def transform_single_row(row: pd.Series | Dict[str, Any], ct) -> np.ndarray:
    """
    Transform a single row (Series or dict-like) with an already-fitted CT or wrapper.
    Only the CT's expected columns are used when available; otherwise falls back to the whitelist.
    Returns [1, D] float32 array.
    """
    num_cols, cat_cols = _extract_mm_cols_from_ct(ct)
    if num_cols is None or cat_cols is None:
        # Fallback to whitelist order
        num_cols, cat_cols = list(TAB_NUM), list(TAB_CAT)

    # Normalize input -> DataFrame with exactly the expected columns
    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    else:
        row_dict = dict(row)

    data = {}
    for c in num_cols:
        v = row_dict.get(c, np.nan)
        try:
            data[c] = pd.to_numeric(pd.Series([v]), errors="coerce")
        except Exception:
            data[c] = pd.Series([np.nan])
    for c in cat_cols:
        v = row_dict.get(c, np.nan)
        data[c] = pd.Series([str(v)])

    dfx = pd.DataFrame(data)
    X = ct.transform(dfx).astype(np.float32, copy=False)
    return X
