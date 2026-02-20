from collections import OrderedDict

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def get_parameters_vision(model) -> list:
    """Return list of numpy arrays in state_dict order (A, B, A, B, ... per layer)."""
    state_dict = get_peft_model_state_dict(model)
    return [v.cpu().numpy() for _, v in state_dict.items()]


def set_parameters_vision(model, parameters: list) -> None:
    """Set PEFT state from list of numpy arrays (same order as get_parameters_vision)."""
    state_dict = get_peft_model_state_dict(model)
    keys = list(state_dict.keys())
    assert len(keys) == len(parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    set_peft_model_state_dict(model, state_dict)
