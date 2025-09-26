# File: scripts/make_sample_from_hdf5.py
import argparse, h5py, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_h5",  default="project/data/dataset.hdf5")
    p.add_argument("--src_csv", default="project/data/metadata.csv")
    p.add_argument("--dst_dir", default="data")
    p.add_argument("--n_per_class", type=int, default=30)  # total ≈ 60 rows
    args = p.parse_args()

    src_h5  = Path(args.src_h5)
    src_csv = Path(args.src_csv)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    assert src_h5.exists(), f"Missing {src_h5}"
    assert src_csv.exists(), f"Missing {src_csv}"

    # Load full CSV
    df = pd.read_csv(src_csv)
    # Coerce label to 0/1 quickly (adjust map to your LABEL_MAP)
    label_col = "histopath_diagnosis"
    labmap = {"benign":0, "Benign":0, 0:0, "malignant":1, "Malignant":1, 1:1}
    y = pd.Series(df[label_col]).map(labmap).astype(int)
    df["label"] = y

    # Sample balanced subset
    idx_ben = df[df["label"]==0].sample(args.n_per_class, random_state=42, replace=False).index
    idx_mal = df[df["label"]==1].sample(args.n_per_class, random_state=42, replace=False).index
    keep = np.sort(np.concatenate([idx_ben.values, idx_mal.values]))

    df_small = df.iloc[keep].copy().reset_index(drop=True)

    # Open HDF5 and copy those frames to a tiny file
    with h5py.File(src_h5, "r") as fin, h5py.File(dst_dir/"dataset_sample.hdf5", "w") as fout:
        imgs = fin["/image"]  # adjust if your key is different
        # Create fixed-size dataset
        n, H, W = len(keep), imgs.shape[1], imgs.shape[2]
        dimg = fout.create_dataset("/image", shape=(n, H, W), dtype=imgs.dtype, compression="gzip", compression_opts=4)
        # Annot/frame ids (string)
        # Keep these as bytes so we can decode later
        ann = fout.create_dataset("/annot_id",  shape=(n,), dtype="S16")
        frm = fout.create_dataset("/frame_num", shape=(n,), dtype="S8")

        for i, k in enumerate(keep):
            dimg[i] = imgs[k]
            a = str(df.iloc[k].get("annot_id", f"A{i}")).encode()
            f = str(df.iloc[k].get("frame_num", f"{i}")).encode()
            ann[i] = a
            frm[i] = f

    # Write aligned CSV
    df_small.to_csv(dst_dir/"metadata_sample.csv", index=False)
    print("Wrote:", dst_dir/"dataset_sample.hdf5", "and", dst_dir/"metadata_sample.csv")

if __name__ == "__main__":
    main()
