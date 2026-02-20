import math

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig,AutoModelForSequenceClassification, AutoConfig

from flwr.common.typing import NDArrays


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )
    
    TASK_TYPE = "SEQ_CLS" if model_cfg.task_type == 'NLU' else "CAUSAL_LM"
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
        "bert": ["query"],
        "roberta": ["query"],
        "roberta-large": ["query"],
        "bert-large-cased":["query"],
        "llama": ["q_proj"],
        "qwen2": ["q_proj"],
        "qwen3": ["q_proj"],
        "mistral": ["q_proj"],
        "open_llama_3b_v2":["q_proj"],
        "open_llama_7b_v2":["q_proj"],
        "Meta-Llama-3-8B":["q_proj"],
        "Llama-3.2-3B":["q_proj"],
        "Qwen3-4B-Instruct-2507":["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        "llama-30b":["q_proj"],
        }
    # Backward-compatible rope_scaling validation for legacy transformers versions (Llama3/Meta-Llama)
    # name_lower = model_cfg.name.lower()
    # rope_scaling_arg = {"type": "linear", "factor": 32.0} if ("meta-llama" in name_lower or "llama-3" in name_lower or "llama-3.2" in name_lower) else None
    if TASK_TYPE == 'CAUSAL_LM':
        if model_cfg.quantization:
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg.name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
    elif TASK_TYPE == 'SEQ_CLS':
        task = model_cfg.task_type
        num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
        if model_cfg.quantization:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_cfg.name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                num_labels = num_labels,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            num_labels = num_labels,
        )
    else:
        raise NotImplementedError("Unsupported Task!")

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_cfg.name.split("/")[1]],  
        lora_dropout=0.075,
        task_type=TASK_TYPE,
    )

    return get_peft_model(model, peft_config)


def set_parameters(model, parameters: NDArrays) -> None:
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]

def save_lora_parameters(model, checkpoint_path: str) -> None:
    state_dict = get_peft_model_state_dict(model)
    torch.save(state_dict, checkpoint_path)  