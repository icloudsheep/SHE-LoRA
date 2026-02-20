#!/usr/bin/env python3
"""
DTD dataset folder setup.
Organizes DTD images/labels into data/dtd/train and data/dtd/val for ImageFolder.

Usage:
  DATA_ROOT=./data python -m vision.scripts.setup_dtd_dataset
  or: python vision/scripts/setup_dtd_dataset.py [data_root]

Download: https://www.robots.ox.ac.uk/~vgg/data/dtd/
After extract: <root>/dtd/images/, <root>/dtd/labels/, train1.txt, val1.txt, etc.
"""
import os
import shutil
import sys


def get_data_root():
    if os.environ.get("DATA_ROOT"):
        return os.environ["DATA_ROOT"].rstrip("/")
    if len(sys.argv) > 1:
        return sys.argv[1].rstrip("/")
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "data")


def setup_dtd_dataset(root_dir, use_split=1):
    """Build train/val dirs from DTD images + labels."""
    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    train_path = os.path.join(labels_dir, "train%d.txt" % use_split)
    val_path = os.path.join(labels_dir, "val%d.txt" % use_split)

    if not os.path.isdir(images_dir):
        print("Error: images dir not found:", images_dir)
        return False
    if not os.path.exists(train_path):
        print("Error: split file not found:", train_path)
        return False
    if not os.path.exists(val_path):
        print("Error: split file not found:", val_path)
        return False

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def copy_from_list(source_list, target_dir):
        copied = 0
        with open(source_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("/")
                if len(parts) != 2:
                    continue
                class_name, filename = parts
                class_dir = os.path.join(target_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(images_dir, line)
                dst = os.path.join(class_dir, filename)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    copied += 1
        return copied

    train_count = copy_from_list(train_path, train_dir)
    val_count = copy_from_list(val_path, val_dir)
    print("Train:", train_count, "Val:", val_count, "->", train_dir, val_dir)
    return True


if __name__ == "__main__":
    data_root = get_data_root()
    dtd_root = os.path.join(data_root, "dtd")
    if not os.path.isdir(dtd_root):
        print("Error: DTD root not found:", dtd_root)
        sys.exit(1)
    ok = setup_dtd_dataset(dtd_root, use_split=1)
    sys.exit(0 if ok else 1)
