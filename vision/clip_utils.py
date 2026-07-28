import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

log = logging.getLogger(__name__)


def _load_clip_from_bin(
    model_name_or_path: str, local_files_only: bool
) -> CLIPModel:
    config = CLIPConfig.from_pretrained(
        model_name_or_path, local_files_only=local_files_only
    )
    clip_model = CLIPModel(config)

    bin_path = os.path.join(model_name_or_path, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Expected local model weights at {bin_path}.")

    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # strict=False: .bin may have deprecated buffers like position_ids not in new model
    clip_model.load_state_dict(state_dict, strict=False)
    return clip_model

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def freeze_vision_only(clip_model: CLIPModel) -> CLIPModel:
    """Freeze all parameters except vision encoder and visual projection."""
    for param in clip_model.parameters():
        param.requires_grad = False
    for param in clip_model.vision_model.parameters():
        param.requires_grad = True
    for param in clip_model.visual_projection.parameters():
        param.requires_grad = True
    return clip_model


def _lora_config_from_cfg(lora_cfg) -> LoraConfig:
    """Build LoraConfig from OmegaConf or dict."""
    if hasattr(lora_cfg, "r"):
        return LoraConfig(
            r=int(lora_cfg.r),
            lora_alpha=int(getattr(lora_cfg, "lora_alpha", lora_cfg.r * 2)),
            target_modules=list(getattr(lora_cfg, "target_modules", ["q_proj", "v_proj"])),
            lora_dropout=float(getattr(lora_cfg, "lora_dropout", 0.0)),
            inference_mode=False,
        )
    if isinstance(lora_cfg, dict):
        return LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),
            inference_mode=False,
        )
    return lora_cfg


def load_clip_processor_and_model(
    model_name_or_path: str,
    lora_config: Optional[Any] = None,
    linearized_lora: bool = False,
    local_files_only: bool = True,
    random_seed: int = 42,
):
    """
    Load CLIP processor and model, optionally wrap vision encoder with LoRA.
    Returns: (processor, clip_model, clip_vision_model, clip_text_model).
    clip_vision_model is the PEFT vision encoder; clip_model.vision_model is set to base for forward.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    if not local_files_only:
        raise ValueError("Vision models must be loaded with local_files_only=True.")
    model_path = Path(model_name_or_path)
    if not model_path.is_dir():
        raise NotADirectoryError(
            f"Vision model path must be an existing local directory: {model_path}"
        )

    processor = CLIPProcessor.from_pretrained(
        str(model_path), local_files_only=True
    )
    # Prefer local safetensors and support a local pytorch_model.bin as fallback.
    try:
        clip_model = CLIPModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_safetensors=True,
        )
    except OSError:
        bin_path = model_path / "pytorch_model.bin"
        if not bin_path.is_file():
            raise
        log.info("Loading local model weights from %s.", bin_path)
        clip_model = _load_clip_from_bin(str(model_path), True)
    clip_model = freeze_vision_only(clip_model)
    clip_vision_model = clip_model.vision_model
    clip_text_model = clip_model.text_model

    if lora_config is not None:
        lora_cfg = _lora_config_from_cfg(lora_config)
        clip_vision_model = get_peft_model(clip_vision_model, lora_cfg)
        if linearized_lora:
            log.warning("linearized_lora requested but not implemented in vision; using standard LoRA.")
        clip_vision_model.print_trainable_parameters()
        # So that get_image_features() uses LoRA weights during training
        clip_model.vision_model = clip_vision_model
    return processor, clip_model, clip_vision_model, clip_text_model


def setup_fabric(cfg: DictConfig):
    """
    Setup Lightning Fabric for logging and device.
    If lightning not available or config minimal, return a minimal object with logger and setup_module/backward.
    """
    try:
        import lightning as L
        from lightning.fabric.loggers.tensorboard import TensorBoardLogger
    except ImportError:
        return _MinimalFabric(cfg)
    model_name = getattr(cfg, "model_name", "clip")
    dataset_name = getattr(cfg, "dataset_name", "vision")
    log_dir = Path("logs") / str(model_name) / str(dataset_name)
    logger = TensorBoardLogger(root_dir=str(log_dir.parent), name=log_dir.name)
    fabric_cfg = getattr(cfg, "fabric", {})
    if isinstance(fabric_cfg, DictConfig):
        fabric_cfg = dict(fabric_cfg)
    fabric = L.Fabric(loggers=logger, **fabric_cfg)
    fabric.launch()
    return fabric


class _MinimalFabric:
    """Minimal Fabric-like object when Lightning is not used: no DDP, just device and logger."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = _MinimalLogger()

    def setup_module(self, module):
        return module.to(self._device)

    def setup_dataloaders(self, *loaders):
        return loaders if len(loaders) > 1 else loaders[0]

    def backward(self, loss):
        loss.backward()

    def launch(self):
        pass


class _MinimalLogger:
    log_dir = "logs/vision"
