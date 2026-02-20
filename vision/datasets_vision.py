import os
import re
from typing import List, Optional, Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN, ImageFolder

# MNIST / CIFAR10 / SVHN class names for CLIP text prompts
MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
SVHN_CLASSES = [str(i) for i in range(10)]

# GTSRB text prompts for CLIP
GTSRB_CLASSES = [
    "red and white circle 20 kph speed limit",
    "red and white circle 30 kph speed limit",
    "red and white circle 50 kph speed limit",
    "red and white circle 60 kph speed limit",
    "red and white circle 70 kph speed limit",
    "red and white circle 80 kph speed limit",
    "end / de-restriction of 80 kph speed limit",
    "red and white circle 100 kph speed limit",
    "red and white circle 120 kph speed limit",
    "red and white circle red car and black car no passing",
    "red and white circle red truck and black car no passing",
    "red and white triangle road intersection warning",
    "white and yellow diamond priority road",
    "red and white upside down triangle yield right-of-way",
    "stop",
    "empty red and white circle",
    "red and white circle no truck entry",
    "red circle with white horizonal stripe no entry",
    "red and white triangle with exclamation mark warning",
    "red and white triangle with black left curve approaching warning",
    "red and white triangle with black right curve approaching warning",
    "red and white triangle with black double curve approaching warning",
    "red and white triangle rough / bumpy road warning",
    "red and white triangle car skidding / slipping warning",
    "red and white triangle with merging / narrow lanes warning",
    "red and white triangle with person digging / construction / road work warning",
    "red and white triangle with traffic light approaching warning",
    "red and white triangle with person walking warning",
    "red and white triangle with child and person walking warning",
    "red and white triangle with bicyle warning",
    "red and white triangle with snowflake / ice warning",
    "red and white triangle with deer warning",
    "white circle with gray strike bar no speed limit",
    "blue circle with white right turn arrow mandatory",
    "blue circle with white left turn arrow mandatory",
    "blue circle with white forward arrow mandatory",
    "blue circle with white forward or right turn arrow mandatory",
    "blue circle with white forward or left turn arrow mandatory",
    "blue circle with white keep right arrow mandatory",
    "blue circle with white keep left arrow mandatory",
    "blue circle with white arrows indicating a traffic circle",
    "white circle with gray strike bar indicating no passing for cars has ended",
    "white circle with gray strike bar indicating no passing for trucks has ended",
]


def _eurosat_pretify(classname: str) -> str:
    """EuroSAT class name to natural language prompt."""
    parts = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", classname)
    out = " ".join(p.lower() for p in parts)
    if out.endswith("al"):
        return out + " area"
    return out


# EuroSAT class name to prompt mapping
EUROSAT_OURS_TO_PROMPT = {
    "annual crop": "annual crop land",
    "forest": "forest",
    "herbaceous vegetation": "brushland or shrubland",
    "highway": "highway or road",
    "industrial area": "industrial buildings or commercial buildings",
    "pasture": "pasture land",
    "permanent crop": "permanent crop land",
    "residential area": "residential buildings or homes or apartments",
    "river": "river",
    "sea lake": "lake or sea",
}


def get_vision_dataset(
    dataset_name: str,
    root: str,
    train: bool,
    input_size: int = 224,
    transform: Optional[transforms.Compose] = None,
    download: bool = True,
) -> Tuple[Dataset, List[str]]:
    """
    Return (dataset, class_names) for a given dataset.
    root: base path (e.g. "data"). Subdirs: dtd/dtd, EuroSAT_splits, gtsrb, svhn, etc.
    """
    name_upper = dataset_name.upper()
    if transform is None:
        if name_upper == "MNIST":
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])

    if name_upper == "MNIST":
        ds = MNIST(root=root, train=train, download=download, transform=transform)
        return ds, MNIST_CLASSES

    if name_upper == "CIFAR10":
        if not train:
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
        ds = CIFAR10(root=root, train=train, download=download, transform=transform)
        return ds, CIFAR10_CLASSES

    if name_upper == "SVHN":
        data_dir = os.path.join(root, "svhn")
        ds = SVHN(
            data_dir, split="train" if train else "test",
            download=download, transform=transform,
        )
        return ds, SVHN_CLASSES

    if name_upper == "GTSRB":
        try:
            from torchvision.datasets import GTSRB as TVGTSRB
        except ImportError:
            raise ImportError(
                "GTSRB requires torchvision >= 0.13. Install with: pip install 'torchvision>=0.13'"
            )
        gtsrb_root = os.path.join(root, "gtsrb")
        ds = TVGTSRB(
            root=gtsrb_root,
            split="train" if train else "test",
            download=download,
            transform=transform,
        )
        return ds, GTSRB_CLASSES

    if name_upper == "DTD":
        subdir = "train" if train else "val"
        dtd_dir = os.path.join(root, "dtd", subdir)
        if not os.path.isdir(dtd_dir):
            raise FileNotFoundError(
                f"DTD not found at {dtd_dir}. "
                "Run: python -m vision.scripts.setup_dtd_dataset (after extracting DTD under data/)."
            )
        ds = ImageFolder(dtd_dir, transform=transform)
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        classes = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
        return ds, classes

    if name_upper == "EURSAT" or name_upper == "EUROSAT":
        if train:
            subdir = "train"
        else:
            subdir = "test"
            for sub in ("test", "validation"):
                cand = os.path.join(root, "EuroSAT_splits", sub)
                if os.path.isdir(cand):
                    subdir = sub
                    break
        eurosat_dir = os.path.join(root, "EuroSAT_splits", subdir)
        if not os.path.isdir(eurosat_dir):
            raise FileNotFoundError(
                f"EuroSAT not found at {eurosat_dir}. "
                "Run: python -m vision.scripts.setup_eurosat_dataset (after placing EuroSAT + CSVs under data/)."
            )
        ds = ImageFolder(eurosat_dir, transform=transform)
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        classes = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
        classes = [_eurosat_pretify(c) for c in classes]
        classes = [EUROSAT_OURS_TO_PROMPT.get(c, c) for c in classes]
        return ds, classes

    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        "Supported: MNIST, CIFAR10, SVHN, GTSRB, DTD, EuroSAT."
    )


def get_vision_dataset_info(dataset_name: str) -> dict:
    """Return info (classes, needs_download, layout) for documentation."""
    name_upper = dataset_name.upper()
    if name_upper in ("MNIST", "CIFAR10"):
        return {"download": True, "layout": "torchvision default"}
    if name_upper == "SVHN":
        return {"download": True, "layout": "data/svhn/"}
    if name_upper == "GTSRB":
        return {"download": True, "layout": "data/gtsrb/ (torchvision GTSRB)"}
    if name_upper == "DTD":
        return {
            "download": False,
            "layout": "data/dtd/train/, data/dtd/val/ (ImageFolder)",
            "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        }
    if name_upper in ("EUROSAT", "EURSAT"):
        return {
            "download": False,
            "layout": "data/EuroSAT_splits/train/, validation/, test/ (ImageFolder)",
        }
    return {}
