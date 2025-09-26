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
    "size_x", "size_y", "size_z",       # lesion dimensions
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
    # Add any known post-biopsy or outcome columns here if present in your CSV
    # e.g., "post_biopsy_treatment", "surgery_outcome", etc.
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
    # Keep track of the columns actually used (for robust single-row transform)
    ct.mm_num_cols = num_cols
    ct.mm_cat_cols = cat_cols
    return ct

def _get_feature_names(ct: ColumnTransformer) -> List[str]:
    """Return expanded feature names after fit."""
    names: List[str] = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if name == "num":
            # StandardScaler keeps column names
            names.extend(list(cols))
        elif name == "cat":
            # Pull names from the OHE inside the pipeline
            ohe = trans.named_steps.get("ohe")
            if hasattr(ohe, "get_feature_names_out"):
                names.extend(list(ohe.get_feature_names_out(cols)))
            else:
                # older sklearn
                names.extend(list(ohe.get_feature_names(cols)))
        else:
            # Fallback: append raw column names
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

def transform_all(df: pd.DataFrame, ct: ColumnTransformer) -> np.ndarray:
    """
    Transform an arbitrary DataFrame with an already-fitted ColumnTransformer.
    Uses the exact numeric/categorical columns ct was fit on (ct.mm_num_cols / ct.mm_cat_cols).
    Returns float32 numpy array [N, D].
    """
    if not hasattr(ct, "mm_num_cols") or not hasattr(ct, "mm_cat_cols"):
        raise AttributeError("ColumnTransformer is missing mm_num_cols/mm_cat_cols. "
                             "Use the ct returned by fit_transform_all().")
    num_cols: List[str] = list(ct.mm_num_cols)
    cat_cols: List[str] = list(ct.mm_cat_cols)

    # Build a projection DataFrame with just the columns ct expects; missing cols become NaN
    data = {}
    for c in num_cols:
        data[c] = pd.to_numeric(df.get(c, pd.Series([np.nan]*len(df))), errors="coerce")
    for c in cat_cols:
        s = df.get(c, pd.Series([np.nan]*len(df)))
        data[c] = s.astype(str)

    dfx = pd.DataFrame(data, index=df.index)
    X = ct.transform(dfx).astype(np.float32, copy=False)
    return X

def transform_single_row(row: pd.Series | Dict[str, Any], ct: ColumnTransformer) -> np.ndarray:
    """
    Transform a single row (Series or dict-like) into a 2D feature array [1, D].
    Only the ct.mm_num_cols and ct.mm_cat_cols are used.
    Missing values are imputed by the pipeline.
    """
    if not hasattr(ct, "mm_num_cols") or not hasattr(ct, "mm_cat_cols"):
        raise AttributeError("ColumnTransformer is missing mm_num_cols/mm_cat_cols. "
                             "Use the ct returned by fit_transform_all().")

    num_cols: List[str] = list(ct.mm_num_cols)
    cat_cols: List[str] = list(ct.mm_cat_cols)

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
