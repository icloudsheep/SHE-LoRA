from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


def get_vision_calibration_loader(
    dataset_name: str = "CIFAR10",
    batch_size: int = 16,
    num_batches: int = 8,
    input_size: int = 224,
    num_workers: int = 0,
    data_root: Optional[str] = None,
) -> DataLoader:
    """
    Return local CIFAR10 or MNIST images for calibration.
    """
    if not data_root or not str(data_root).strip():
        raise ValueError("A local vision dataset root is required for calibration.")
    root = str(data_root)
    n_total = num_batches * batch_size
    if dataset_name.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        ds = MNIST(root=root, train=True, download=False, transform=transform)
    elif dataset_name.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        ds = CIFAR10(root=root, train=True, download=False, transform=transform)
    else:
        raise ValueError(
            "Standalone vision calibration supports local MNIST or CIFAR10 only."
        )
    from torch.utils.data import Subset

    ds = Subset(ds, range(min(n_total, len(ds))))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def get_vision_calibration_loader_from_loader(
    train_loader: DataLoader,
    num_batches: int = 8,
) -> DataLoader:
    """Build calibration loader from an existing train DataLoader (same dataset as client)."""
    batches = []
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        batches.append(images)
    if not batches:
        return train_loader
    from torch.utils.data import TensorDataset
    data = torch.cat(batches, dim=0)
    return DataLoader(TensorDataset(data), batch_size=train_loader.batch_size, shuffle=False)
