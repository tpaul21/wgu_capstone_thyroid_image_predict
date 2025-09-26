# project/self_check.py
from project.config import HDF5_PATH, HDF5_IMAGES_DATASET, HDF5_METADATA_PATH, HDF5_LABEL_COLUMN, LABEL_MAP, set_seed
from project.data_hdf5 import _read_metadata_df, _to_pil_rgb, make_splits_and_loaders_mm
from project.model import build_multimodal_resnet
from project.engine import train
import h5py, numpy as np

def main():
    set_seed()
    # HDF5 reachable & keys exist
    with h5py.File(HDF5_PATH, "r") as f:
        assert HDF5_IMAGES_DATASET in f, "Images dataset path wrong"
        assert HDF5_METADATA_PATH in f, "Metadata path wrong"
        df = _read_metadata_df(f)
    assert HDF5_LABEL_COLUMN in df.columns, "Label column missing"
    unmapped = set(df[HDF5_LABEL_COLUMN].astype(str).unique()) - set(map(str, LABEL_MAP.keys()))
    assert not unmapped, f"Unmapped labels: {unmapped}"

    # Dataloaders + one batch
    train_dl, val_dl, test_dl, ct, feat_names, tab_dim = make_splits_and_loaders_mm(batch_size=2)
    batch = next(iter(train_dl))
    assert len(batch) == 3, "Expect (x_img, x_tab, y)"
    x_img, x_tab, y = batch
    assert x_img.ndim == 4 and x_tab.ndim == 2 and y.ndim == 1, "Bad batch shapes"

    # Forward pass
    model = build_multimodal_resnet(tab_dim)
    logits = model(x_img, x_tab).squeeze(1)
    assert logits.shape[0] == x_img.shape[0], "Logit batch mismatch"
    print("✅ self_check passed")

if __name__ == "__main__":
    main()
