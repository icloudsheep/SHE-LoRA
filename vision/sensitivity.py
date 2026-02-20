import torch
import torch.nn as nn
from tqdm import tqdm
from peft.tuners.lora import LoraLayer

from .layerwrapper import WrappedVisionLinear


def find_lora_layers_vision(module: nn.Module, name: str = "") -> dict:
    """Find all LoRA layers in a CLIP vision encoder (or any nn.Module)."""
    if isinstance(module, LoraLayer):
        return {name: module}
    res = {}
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        res.update(find_lora_layers_vision(child, full_name))
    return res


def evaluate_lora_importance_vision(
    vision_model: nn.Module,
    dataloader,
    device: torch.device,
    nsamples: int = 128,
) -> tuple:
    vision_model.eval()
    # Resolve the actual vision encoder if wrapped (e.g. PeftModel.base_model)
    model = getattr(vision_model, "base_model", vision_model)
    model = getattr(model, "model", model)

    layers = list(model.modules())
    # Find LoRA layers with their full names (match state_dict key prefix)
    lora_layers = find_lora_layers_vision(model, "")
    if not lora_layers:
        return {}, {}, {}

    importance_scores = {}
    element_importance = {}
    negotiation_scores = {}

    wrapped = {n: WrappedVisionLinear(m, layer_name=n) for n, m in lora_layers.items()}
    handles = []
    for n, m in lora_layers.items():
        def _make_hook(name):
            def _hook(_, inp, out):
                wrapped[name].add_batch(inp, out)
            return _hook
        handles.append(m.register_forward_hook(_make_hook(n)))

    n_used = 0
    with torch.no_grad():
        for batch in dataloader:
            if n_used >= nsamples:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            if hasattr(model, "forward"):
                _ = model(pixel_values=images)
            else:
                _ = model(images)
            n_used += images.shape[0]

    for h in handles:
        h.remove()

    for layer_name, layer_module in lora_layers.items():
        if not (hasattr(layer_module, "lora_A") and "default" in layer_module.lora_A):
            continue
        lora_a = layer_module.lora_A["default"].weight.data
        wr = wrapped[layer_name]
        x_norm = torch.sqrt(wr.scaler_row.reshape(1, -1).to(lora_a.device) + 1e-8)
        W_metric = (torch.abs(lora_a * x_norm)).sum(dim=0)
        W_metric_elem = torch.abs(lora_a * x_norm)
        importance_scores[layer_name] = W_metric.detach().cpu().numpy()
        element_importance[layer_name] = W_metric_elem.detach().cpu().numpy()
        scores_cols = [
            {"line": int(j), "score": float(importance_scores[layer_name][j])}
            for j in range(importance_scores[layer_name].shape[0])
        ]
        scores_cols.sort(key=lambda x: x["score"], reverse=True)
        negotiation_scores[layer_name] = scores_cols

    return importance_scores, element_importance, negotiation_scores
