#!/usr/bin/env python3
"""
EuroSAT dataset folder setup.
Organizes raw EuroSAT dir + train/validation/test CSV into data/EuroSAT_splits/{train,validation,test}.

Usage:
  DATA_ROOT=./data python -m vision.scripts.setup_eurosat_dataset
  or: python -m vision.scripts.setup_eurosat_dataset [data_root]

Requires: train.csv, validation.csv, test.csv (columns ClassName, Filename) in base dir.
"""
import os
import shutil
import sys
from pathlib import Path


def get_data_root() -> str:
    if os.environ.get("DATA_ROOT"):
        return os.environ["DATA_ROOT"].rstrip("/")
    if len(sys.argv) > 1:
        return sys.argv[1].rstrip("/")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


def setup_eurosat_dataset(base_data_dir: str, target_data_dir: str) -> bool:
    """Organize EuroSAT into train/validation/test subdirs from CSV splits."""
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas required. pip install pandas")
        return False

    base = Path(base_data_dir)
    target = Path(target_data_dir)
    csv_files = {
        "train": base / "train.csv",
        "validation": base / "validation.csv",
        "test": base / "test.csv",
    }

    if not base.is_dir():
        print(f"Error: base dir not found: {base}")
        return False

    for split_name, csv_path in csv_files.items():
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {split_name}")
            continue
        print(f"Processing {split_name}...")
        df = pd.read_csv(csv_path)
        if "ClassName" not in df.columns or "Filename" not in df.columns:
            print(f"Error: CSV must have ClassName and Filename columns: {csv_path}")
            return False
        classes = df["ClassName"].unique()
        for c in classes:
            (target / split_name / c).mkdir(parents=True, exist_ok=True)
        count = 0
        for _, row in df.iterrows():
            src = base / row["Filename"]
            dst = target / split_name / row["Filename"]
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                count += 1
            else:
                print(f"Warning: missing {src}")
        print(f"  {split_name}: {count} files")

    print(f"Done. EuroSAT_splits at: {target}")
    return True


if __name__ == "__main__":
    data_root = get_data_root()
    base_dir = os.path.join(data_root, "EuroSAT")
    target_dir = os.path.join(data_root, "EuroSAT_splits")
    if not os.path.isdir(base_dir):
        print(f"Error: EuroSAT base not found: {base_dir}")
        print("Download EuroSAT and place train.csv, validation.csv, test.csv in it.")
        sys.exit(1)
    success = setup_eurosat_dataset(base_dir, target_dir)
    sys.exit(0 if success else 1)
