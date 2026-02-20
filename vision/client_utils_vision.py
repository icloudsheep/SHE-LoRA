from collections import OrderedDict

from peft import get_peft_model_state_dict

from .models_vision import get_parameters_vision
from flowertune_llm.she.ckks_client import exchange_columns, encrypt_cipher_list
from flowertune_llm.utils.client_utils import get_plain_cipher


def _find_loraAid_and_enc_lines_vision(peft_model, enc_lines: dict):
    """Enc_lines: layer_name -> list of column indices (plain). Returns order aligned with state_dict."""
    import logging
    log = logging.getLogger(__name__)
    
    state_dict = get_peft_model_state_dict(peft_model)
    lora_a_keys = [k for k in state_dict.keys() if "lora_A" in k and "weight" in k]
    # Layer name: strip .lora_A.default.weight
    layer_names = [
        k.replace(".lora_A.default.weight", "").replace(".lora_A.weight", "")
        for k in lora_a_keys
    ]
    index_as = [i for i, k in enumerate(state_dict.keys()) if "lora_A" in k and "weight" in k]
    
    # enc_lines keys should already match layer_names (caller normalized them)
    enc_a_lines = []
    matched = 0
    for ln in layer_names:
        cols = enc_lines.get(ln, [])
        if cols:
            matched += 1
        enc_a_lines.append(cols[:])  # copy
    
    log.debug("Matched %d/%d layers with enc_lines", matched, len(layer_names))
    return index_as, enc_a_lines


def set_enc_lines_to_client_plain(enc_lines: dict, he_budget: int) -> dict:
    """Cut global enc_lines (plain indices) to client budget. enc_lines: layer -> list of int."""
    return {layer: list(cols[:he_budget]) for layer, cols in enc_lines.items()}


def handle_parameters_to_server_vision(model, enc_lines: dict, he_budget: int):
    """
    Same as flowertune_llm.utils.client_utils.handle_parameters_to_server but uses
    get_parameters_vision and enc_lines as plain column indices (layer -> list of int).
    Returns (plain_ndarrays, cipher_bytes).
    """
    import logging
    log = logging.getLogger(__name__)
    
    def normalize_name(name):
        """Strip common prefixes to get core layer name."""
        for prefix in ["base_model.model.", "model."]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name
    
    parameters = get_parameters_vision(model)
    # enc_lines order must match state_dict lora_A order
    state_dict = get_peft_model_state_dict(model)
    lora_a_keys = [k for k in state_dict.keys() if "lora_A" in k and "weight" in k]
    layer_names = [
        k.replace(".lora_A.default.weight", "").replace(".lora_A.weight", "")
        for k in lora_a_keys
    ]
    
    # Normalize enc_lines keys for matching
    enc_lines_normalized = {normalize_name(k): v for k, v in enc_lines.items()}
    
    # Build enc_lines_ordered matching state_dict order
    enc_lines_ordered = OrderedDict()
    for ln in layer_names:
        ln_norm = normalize_name(ln)
        cols = enc_lines_normalized.get(ln_norm, [])[:he_budget]
        enc_lines_ordered[ln] = cols
    
    index_as, enc_a_lines = _find_loraAid_and_enc_lines_vision(model, enc_lines_ordered)
    
    # Debug: check enc_a_lines
    num_enc_cols = len(enc_a_lines[0]) if enc_a_lines else 0
    log.debug("enc_a_lines: %d layers, first layer has %d columns to encrypt", len(enc_a_lines), num_enc_cols)
    
    plain, cipher_list = get_plain_cipher(
        index_as=index_as, enc_a_lines=enc_a_lines, parameters=parameters
    )
    
    # Debug: check cipher_list
    cipher_shape = cipher_list[0].shape if cipher_list else None
    log.debug("cipher_list: %d layers, first layer shape: %s", len(cipher_list), cipher_shape)
    
    cipher_bytes = encrypt_cipher_list(cipher_list)
    return plain, cipher_bytes


def apply_parameters_vision_from_fusion(model, fused_parameters: list) -> None:
    """Apply fused LoRA parameters (list of A,B,A,B,...) to vision PEFT model."""
    from .models_vision import set_parameters_vision
    set_parameters_vision(model, fused_parameters)
