import torch
import torch.nn as nn
from peft.tuners.lora.layer import Linear as LoraLinear


class WrappedVisionLinear:
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = layer.weight.device if hasattr(layer, "weight") else next(layer.parameters()).device
        # LoRA A shape (r, in_features); PEFT uses .lora_A["default"].weight
        try:
            w = layer.lora_A["default"].weight
        except (KeyError, TypeError, AttributeError):
            w = getattr(layer, "weight", next(layer.parameters()))
        self.rows, self.columns = w.shape[0], w.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if inp[0].dim() == 3:
            inp = inp[0].reshape(-1, inp[0].shape[-1])
        else:
            inp = inp[0]
        tmp = inp.shape[0]
        inp = inp.t().float()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
