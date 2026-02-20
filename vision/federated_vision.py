import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Subset
from torchvision import transforms
from tqdm import tqdm

from vision.datasets_vision import get_vision_dataset

log = logging.getLogger(__name__)

# Re-export for backward compatibility
from vision.datasets_vision import CIFAR10_CLASSES, MNIST_CLASSES  # noqa: F401


class FederatedClient:
    """Federated client: local CLIP vision LoRA training and evaluation."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        vision_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        text_embeds: torch.Tensor,
        dataset_name: str,
        q: int = 1,
    ):
        self.client_id = client_id
        self.model = model
        self.vision_model = vision_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.text_embeds = text_embeds
        self.dataset_name = dataset_name
        self.step_count = 0
        self.q = q

    def train_local(self, num_steps: int, fabric: Any) -> float:
        self.vision_model.train()
        device = next(self.vision_model.parameters()).device
        total_loss = 0.0
        num_batches = 0
        steps_taken = 0
        pbar = tqdm(total=num_steps, desc=f"Client {self.client_id}", unit="step", ncols=100)
        while steps_taken < num_steps:
            for batch in self.train_loader:
                if steps_taken >= num_steps:
                    break
                self.optimizer.zero_grad()
                images, labels = batch[0].to(device), batch[1].to(device)
                image_embeds = self.model.get_image_features(pixel_values=images)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = self.text_embeds.to(image_embeds.device)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp().item()
                logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
                logits_per_image = logits_per_text.t()
                loss = F.cross_entropy(logits_per_image, labels)
                if hasattr(fabric, "backward"):
                    fabric.backward(loss)
                else:
                    loss.backward()
                self.optimizer.step()
                if hasattr(self.lr_scheduler, "step"):
                    self.lr_scheduler.step(self.step_count)
                total_loss += loss.item()
                num_batches += 1
                steps_taken += 1
                self.step_count += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, model: nn.Module, fabric: Any) -> float:
        model.eval()
        correct, total = 0, 0
        device = next(self.vision_model.parameters()).device
        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch[0].to(device), batch[1].to(device)
                image_embeds = self.model.get_image_features(pixel_values=images)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = self.text_embeds.to(device)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp().item()
                logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
                logits_per_image = logits_per_text.t()
                predictions = logits_per_image.argmax(dim=1)
                correct += predictions.eq(labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0


class FederatedServer:
    """Server: average LoRA parameters."""

    def __init__(
        self,
        num_clients: int,
        aggregation_method: str = "average",
        q: int = 1,
        cfg: Any = None,
    ):
        self.num_clients = num_clients
        self.aggregation_method = aggregation_method
        self.q = q
        self.cfg = cfg

    def aggregate_parameters(
        self, client_parameters: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if self.aggregation_method != "average":
            log.warning("Only average aggregation is implemented; using average.")
        aggregated = {}
        param_names = list(client_parameters[0].keys())
        for name in param_names:
            stacked = torch.stack([cp[name] for cp in client_parameters])
            aggregated[name] = stacked.mean(dim=0)
        return aggregated


def extract_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract LoRA A/B parameters from PEFT model (name -> tensor)."""
    lora_params = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for key, param in module.named_parameters():
                if "lora_A" in key or "lora_B" in key:
                    full_name = f"{name}.{key}"
                    lora_params[full_name] = param.data.clone()
    return lora_params


def apply_lora_parameters(
    model: nn.Module, lora_params: Dict[str, torch.Tensor]
) -> None:
    """Apply LoRA parameters to model."""
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for key, param in module.named_parameters():
                if "lora_A" in key or "lora_B" in key:
                    full_name = f"{name}.{key}"
                    if full_name in lora_params:
                        param.data.copy_(lora_params[full_name])
                        param.requires_grad = True


def setup_client_dataloaders(
    dataset_name: str,
    batch_size: int,
    input_size: int = 224,
    num_samples: Optional[int] = None,
    cfg: Any = None,
    data_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create train and test dataloaders (MNIST, CIFAR10, DTD, EuroSAT, GTSRB, SVHN).
    dataset_name: one of "MNIST", "CIFAR10", "DTD", "EuroSAT", "GTSRB", "SVHN".
    Returns (train_loader, test_loader, class_names).
    """
    root = data_root or "./data"
    num_workers = getattr(cfg, "num_workers", 0) if cfg else 0
    download = getattr(cfg, "download_datasets", True) if cfg else True

    train_ds, classes = get_vision_dataset(
        dataset_name, root, train=True,
        input_size=input_size, download=download,
    )
    test_ds, _ = get_vision_dataset(
        dataset_name, root, train=False,
        input_size=input_size, download=download,
    )

    if num_samples is not None and num_samples > 0:
        train_ds = Subset(train_ds, range(min(num_samples, len(train_ds))))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=min(256, len(test_ds)), shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, classes


def create_text_embeddings(
    classes: List[str],
    clip_model: nn.Module,
    processor: Any,
) -> torch.Tensor:
    """Create text embeddings for class names (e.g. 'a photo of a {class}')."""
    text = [f"a photo of a {c}" for c in classes]
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**inputs)
    return text_embeds


class CosineAnnealingWithWarmup:
    """Minimal cosine annealing with linear warmup (no peta.optim dependency)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: float,
        warmup_steps: int = 0,
        max_steps: int = 1000,
    ):
        self.optimizer = optimizer
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def step(self, step: int) -> None:
        if step < self.warmup_steps:
            lr = self.base_lrs * (step + 1) / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr = self.base_lrs * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
