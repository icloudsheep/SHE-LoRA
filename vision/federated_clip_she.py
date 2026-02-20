import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Ensure project root and vision are on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from vision.clip_utils import load_clip_processor_and_model, setup_fabric
from vision.federated_vision import (
    FederatedClient,
    FederatedServer,
    extract_lora_parameters,
    apply_lora_parameters,
    setup_client_dataloaders,
    create_text_embeddings,
    CosineAnnealingWithWarmup,
)
from vision.sensitivity import evaluate_lora_importance_vision
from vision.calibration_data import get_vision_calibration_loader_from_loader
from vision.client_utils_vision import (
    set_enc_lines_to_client_plain,
    handle_parameters_to_server_vision,
    apply_parameters_vision_from_fusion,
)
from vision.models_vision import get_parameters_vision
from peft import get_peft_model_state_dict

log = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_float32_matmul_precision("medium")


def _setup_logging() -> None:
    """Setup colored logging (no peta dependency)."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
        stream=sys.stdout,
    )


def _negotiation_round(
    clients: List[FederatedClient],
    he_budgets: List[int],
    lora_rank: int,
    device: torch.device,
    nsamples: int = 64,
) -> Dict[str, List[int]]:
    """Run sensitivity on each client, then server negotiation -> enc_lines."""
    all_scores = {}
    for i, client in enumerate(clients):
        cal_loader = get_vision_calibration_loader_from_loader(
            client.train_loader,
            num_batches=max(1, nsamples // (client.train_loader.batch_size or 32)),
        )
        _, _, negotiation_scores = evaluate_lora_importance_vision(
            client.vision_model, cal_loader, device, nsamples=nsamples
        )
        all_scores[i] = negotiation_scores

    try:
        from flowertune_llm.strategy.negotiation import get_nego_enclines_bayesian
        from types import SimpleNamespace
        mock_clients = [
            SimpleNamespace(
                cid=str(i),
                properties={"he_budget": he_budgets[i] if i < len(he_budgets) else he_budgets[0]},
            )
            for i in range(len(clients))
        ]
        enc_lines = get_nego_enclines_bayesian(mock_clients, all_scores)
    except Exception as e:
        log.warning("Negotiation failed (%s), using empty enc_lines.", e)
        enc_lines = {}
        if all_scores:
            for layer in next(iter(all_scores.values())):
                enc_lines[layer] = []
    return enc_lines


class FederatedCLIPSHETrainer:
    """Federated CLIP trainer with SHE-LoRA: sensitivity + optional HE. Self-contained."""

    def __init__(self, cfg: DictConfig, fabric: Any, use_she: bool = True):
        self.cfg = cfg
        self.fabric = fabric
        self.use_she = use_she
        self.clients: List[FederatedClient] = []
        self.server: Optional[FederatedServer] = None
        self.enc_lines: Dict[str, List[int]] = {}
        he_budgets = getattr(cfg.federated, "he_budgets", [4] * len(cfg.federated.clients))
        if len(he_budgets) < len(cfg.federated.clients):
            he_budgets = he_budgets + [he_budgets[-1]] * (
                len(cfg.federated.clients) - len(he_budgets)
            )
        self.he_budgets = he_budgets

        model_name = getattr(cfg.model, "model_name_or_path", "openai/clip-vit-base-patch16")
        lora_cfg = getattr(cfg, "lora_config", {"r": 8, "lora_alpha": 16})
        log.info("Creating global CLIP model: %s", model_name)
        self.processor, self.clip_model, self._vision, self._text = load_clip_processor_and_model(
            model_name,
            lora_cfg,
            linearized_lora=getattr(cfg, "linearized_lora", False),
            random_seed=getattr(cfg, "seed", 42),
        )

    def setup_clients(self) -> None:
        log.info("Setting up federated clients")
        model_name = getattr(self.cfg.model, "model_name_or_path", "openai/clip-vit-base-patch16")
        input_size = getattr(self.cfg.model, "input_size", 224)
        lora_cfg = getattr(self.cfg, "lora_config", {"r": 8, "lora_alpha": 16})
        for idx, client_cfg in enumerate(self.cfg.federated.clients):
            client_id = client_cfg.id
            dataset_name = getattr(client_cfg, "dataset", "CIFAR10")
            num_samples = getattr(client_cfg, "num_samples", None)
            log.info("Client %s dataset %s", client_id, dataset_name)

            _, client_model, client_vision_model, _ = load_clip_processor_and_model(
                model_name,
                lora_cfg,
                linearized_lora=getattr(self.cfg, "linearized_lora", False),
                random_seed=self.cfg.seed + client_id,
            )
            data_root = getattr(self.cfg, "data_root", None) or getattr(
                getattr(self.cfg, "federated", None), "data_root", None
            ) or "./data"
            train_loader, test_loader, classes = setup_client_dataloaders(
                dataset_name=dataset_name,
                batch_size=self.cfg.batch_size,
                input_size=input_size,
                num_samples=num_samples,
                cfg=self.cfg,
                data_root=data_root,
            )
            text_embeds = create_text_embeddings(
                classes, self.clip_model, self.processor
            )
            optimizer = torch.optim.AdamW(
                [p for p in client_vision_model.parameters() if p.requires_grad],
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
            lr_scheduler = CosineAnnealingWithWarmup(
                optimizer,
                base_lrs=self.cfg.learning_rate,
                warmup_steps=self.cfg.warmup_steps,
                max_steps=self.cfg.federated.local_steps * self.cfg.federated.num_rounds,
            )
            if hasattr(self.fabric, "setup_module"):
                self.fabric.setup_module(client_model.visual_projection)
                client_vision_model = self.fabric.setup_module(client_vision_model)
                train_loader, test_loader = self.fabric.setup_dataloaders(
                    train_loader, test_loader
                )
            else:
                dev = next(client_vision_model.parameters()).device
                client_vision_model = client_vision_model.to(dev)

            client = FederatedClient(
                client_id=client_id,
                model=client_model,
                vision_model=client_vision_model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                text_embeds=text_embeds,
                dataset_name=dataset_name,
                q=getattr(self.cfg.federated, "q", 1),
            )
            self.clients.append(client)

    def setup_server(self) -> None:
        self.server = FederatedServer(
            num_clients=len(self.clients),
            aggregation_method=getattr(
                self.cfg.federated, "aggregation_method", "average"
            ),
            q=getattr(self.cfg.federated, "q", 1),
            cfg=self.cfg,
        )

    def train_round(self, round_idx: int) -> None:
        log.info("Federated Round %s / %s", round_idx + 1, self.cfg.federated.num_rounds)

        # Step 1: All clients train (no encryption yet)
        raw_lora_params = []
        num_examples_list = []
        for client in self.clients:
            log.info("Training Client %s ...", client.client_id)
            _ = client.train_local(
                num_steps=self.cfg.federated.local_steps, fabric=self.fabric
            )
            try:
                n_ex = len(client.train_loader.dataset)
            except Exception:
                n_ex = 1000
            num_examples_list.append(n_ex)
            lora_params = extract_lora_parameters(client.vision_model)
            raw_lora_params.append((lora_params, n_ex))
            log.info("Client %s training done, extracted %d LoRA params.", client.client_id, len(lora_params))

        # Step 2: Encryption (if SHE enabled) - after all clients trained
        if self.use_she and self.enc_lines:
            # Debug: show enc_lines sample
            # sample_layer = next(iter(self.enc_lines.keys())) if self.enc_lines else None
            # sample_cols = self.enc_lines.get(sample_layer, [])[:5] if sample_layer else []
            # log.info("Encrypting parameters with SHE (enc_lines sample: %s -> %s)...", sample_layer, sample_cols)
            log.info("Encrypting parameters with SHE...")
            client_parameters = []
            for i, (lora_params, n_ex) in enumerate(raw_lora_params):
                he_budget = self.he_budgets[i] if i < len(self.he_budgets) else self.he_budgets[0]
                client_enc = set_enc_lines_to_client_plain(self.enc_lines, he_budget)
                plain_list, cipher_bytes = handle_parameters_to_server_vision(
                    self.clients[i].vision_model, client_enc, he_budget
                )
                client_parameters.append((plain_list, cipher_bytes, n_ex))
                log.info("Client %s encryption done.", i)
        else:
            client_parameters = [(lp, None, n) for lp, n in raw_lora_params]

        # Step 3: Aggregation
        log.info("Aggregation started")
        if self.use_she and self.enc_lines and any(c[1] is not None for c in client_parameters):
            aggregated_params = self._aggregate_with_she(client_parameters, num_examples_list)
            for client in self.clients:
                apply_parameters_vision_from_fusion(client.vision_model, aggregated_params)
        else:
            lora_list = [c[0] for c in client_parameters]
            aggregated = self.server.aggregate_parameters(lora_list)
            for client in self.clients:
                apply_lora_parameters(client.vision_model, aggregated)
        log.info("Aggregation done")

        # Step 4: Evaluation
        eval_every = max(1, getattr(self.cfg.federated, "evaluate_every_n_rounds", 1))
        if (round_idx + 1) % eval_every == 0:
            for client in self.clients:
                acc = client.evaluate(client.vision_model, fabric=self.fabric)
                log.info("Round %d - Client %s test acc: %.4f", round_idx + 1, client.client_id, acc)

    def _aggregate_with_she(
        self, client_parameters: list, num_examples_list: list
    ) -> list:
        """
        SHE aggregation following SHE-LoRA protocol (FLORA style):
        
        Key insight from paper Section 3.3:
        - Plain part: aggregate B_plain × A_plain (the product, not separate A/B)
        - Cipher part: aggregate B × cipher_A (in encrypted domain)
        - Then merge via SVD reparameterization
        
        client_parameters: [(plain_list, cipher_bytes, n_ex), ...]
        Returns: list of aggregated LoRA params [A, B, A, B, ...]
        """
        import numpy as np
        import pickle
        from flowertune_llm.she.ckks_server import aggregate_ckks_tensors
        from flowertune_llm.she.ckks_client import _get_ckks_context
        import tenseal as ts

        total = sum(num_examples_list)
        scales = [n / total for n in num_examples_list]
        lora_r = int(getattr(self.cfg.lora_config, "r", 8))
        context = _get_ckks_context()

        # 1. Aggregate plain part: B_plain × A_plain products (FLORA style)
        # plain_list structure: [A0, B0, A1, B1, ...] where A has encrypted cols zeroed
        # We compute BA = B @ A_plain for each layer, then weighted average
        plain_ba_agg = None  # List of aggregated BA matrices per layer
        for (plain_list, _, _), scale in zip(client_parameters, scales):
            client_ba = []
            num_layers = len(plain_list) // 2
            for layer_idx in range(num_layers):
                A_plain = np.array(plain_list[layer_idx * 2])      # (r, in_features)
                B = np.array(plain_list[layer_idx * 2 + 1])        # (out_features, r)
                BA = B @ A_plain  # (out_features, in_features)
                client_ba.append(BA * scale)
            
            if plain_ba_agg is None:
                plain_ba_agg = client_ba
            else:
                plain_ba_agg = [agg + ba for agg, ba in zip(plain_ba_agg, client_ba)]
        
        log.info("Aggregate plain")
        # 2. Server-side: compute B × cipher_A for each client, then aggregate
        # plain_list structure: [A0, B0, A1, B1, ...] where A has encrypted cols zeroed
        # cipher_bytes: pickled list of [layer][col] encrypted A columns
        
        cipher_lists = []
        for (_, cipher_bytes, _) in client_parameters:
            if cipher_bytes is not None:
                cipher_lists.append(pickle.loads(cipher_bytes))
            else:
                cipher_lists.append(None)

        num_layers = len(cipher_lists[0]) if cipher_lists[0] else 0
        
        if num_layers == 0:
            # No cipher: just SVD decompose the plain BA aggregation
            aggregated_params = []
            for ba_matrix in plain_ba_agg:
                U, s, Vh = np.linalg.svd(ba_matrix, full_matrices=False)
                B_agg = U[:, :lora_r] @ np.diag(s[:lora_r])
                A_agg = Vh[:lora_r, :]
                aggregated_params.append(A_agg)
                aggregated_params.append(B_agg)
            return aggregated_params

        # 2. Server-side: compute B × cipher_A
        # Prepare plain_B dict and cipherA dict
        plain_B_dict = {}
        cipherA_dict = {}
        for client_idx, ((plain_list, cipher_bytes, _), cl) in enumerate(
            zip(client_parameters, cipher_lists)
        ):
            if cl is None:
                continue
            client_key = f"client_{client_idx}"
            # Extract B matrices for all layers: plain_list is [A0, B0, A1, B1, ...]
            B_list = [np.array(plain_list[layer_idx * 2 + 1]) 
                      for layer_idx in range(num_layers)]
            plain_B_dict[client_key] = B_list
            cipherA_dict[client_key] = pickle.dumps(cl)
        
        # Compute B × cipher_A (use serial version to avoid fork() deadlock in multi-threaded env)
        if plain_B_dict:
            log.info("Aggregate cipher")
            cipher_ba_bytes = self._compute_b_cipher_a(plain_B_dict, cipherA_dict, context)
            cipher_ba_agg = pickle.loads(cipher_ba_bytes)
        else:
            cipher_ba_agg = [[] for _ in range(num_layers)]

        # 3. Decode and merge plain_ba_agg with cipher_ba_agg
        log.info("Reparameterize")
        aggregated_params = self._fusion_plain_cipher(
            plain_ba_agg, cipher_ba_agg, lora_r
        )
        return aggregated_params

    def _compute_b_cipher_a(
        self, plain_B: dict, cipherA: dict, context
    ) -> bytes:
        """
        Compute B × cipher_A for all clients (server-side computation).
        Optimized version: avoid repeated context loading and tensor creation.
        
        Args:
            plain_B: {client_key: [B_layer0, B_layer1, ...]} numpy arrays
            cipherA: {client_key: pickled [layer][col] encrypted A columns}
            context: CKKS context (reused, not reloaded)
        
        Returns:
            pickled list of aggregated B×A results per layer
        """
        import numpy as np
        import pickle
        import tenseal as ts

        # Parse all cipher data first
        cipher_data = {k: pickle.loads(v) for k, v in cipherA.items()}
        
        # Get layer count
        first_client = next(iter(plain_B.keys()))
        layer_num = len(plain_B[first_client])
        
        # Pre-compute B transpose tensors for all clients and layers (avoid repeated creation)
        B_tensors = {}
        for client_key, B_list in plain_B.items():
            B_tensors[client_key] = []
            for B_matrix in B_list:
                # B: (out_features, r) -> B.T: (r, out_features)
                B_T = np.array(B_matrix).T.tolist()
                B_tensors[client_key].append(ts.plain_tensor(B_T))
        
        # Compute B × cipher_A for each layer and aggregate
        final_res = []
        for layer_idx in range(layer_num):
            # Collect all B×A results for this layer from all clients
            layer_ba_encrypted = []  # List of lists: [[client0_col0, col1, ...], [client1_col0, ...]]
            
            for client_key in plain_B.keys():
                cipher_cols = cipher_data[client_key][layer_idx] if layer_idx < len(cipher_data[client_key]) else []
                if not cipher_cols:
                    continue
                    
                B_tensor = B_tensors[client_key][layer_idx]
                client_ba_cols = []
                
                for vec_a_bytes in cipher_cols:
                    enc_col_a = ts.ckks_vector_from(context, vec_a_bytes)
                    # enc_col_a: (r,), B_tensor: (r, out_features) -> result: (out_features,)
                    ba_result = enc_col_a.mm(B_tensor)
                    client_ba_cols.append(ba_result)  # Keep as CKKS vector, not serialized yet
                
                layer_ba_encrypted.append(client_ba_cols)
            
            # Aggregate CKKS vectors directly (avoid serialize/deserialize overhead)
            if not layer_ba_encrypted:
                final_res.append([])
                continue
            
            # Number of columns (should be same for all clients)
            num_cols = max(len(cols) for cols in layer_ba_encrypted)
            layer_agg = []
            
            for col_idx in range(num_cols):
                result = None
                num_tensors = 0
                for client_cols in layer_ba_encrypted:
                    if col_idx < len(client_cols):
                        if result is None:
                            result = client_cols[col_idx]
                        else:
                            result = result + client_cols[col_idx]  # CKKS addition
                        num_tensors += 1
                
                if result is not None and num_tensors > 0:
                    layer_agg.append({
                        'par': result.serialize(),
                        'num': num_tensors
                    })
            
            final_res.append(layer_agg)
        
        return pickle.dumps(final_res)

    def _fusion_plain_cipher(
        self, plain_ba_agg: list, cipher_ba_agg: list, max_rank: int
    ) -> list:
        """
        Merge plain BA aggregation and decrypted cipher BA (Eq. (4) / Section 3.4).
        Returns: [A0, B0, A1, B1, ...] in layer order.
        """
        import numpy as np
        import tenseal as ts
        from flowertune_llm.she.ckks_client import exchange_columns, _get_ckks_context
        from concurrent.futures import ThreadPoolExecutor

        context = _get_ckks_context()

        # Get enc_lines in state_dict order (same as when encrypting)
        state_dict = get_peft_model_state_dict(self.clients[0].vision_model)
        lora_a_keys = [k for k in state_dict.keys() if "lora_A" in k and "weight" in k]
        layer_names = [
            k.replace(".lora_A.default.weight", "").replace(".lora_A.weight", "")
            for k in lora_a_keys
        ]
        
        # Normalize layer names to match enc_lines keys
        def normalize_name(name):
            for prefix in ["base_model.model.", "model."]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
            return name
        
        enc_a_lines = [self.enc_lines.get(normalize_name(ln), []) for ln in layer_names]

        # Step 1: Batch decrypt all cipher columns across all layers (parallelized)
        def decrypt_single(enc_ba):
            """Decrypt a single CKKS vector."""
            if isinstance(enc_ba, dict):
                ckks_tensor = ts.ckks_vector_from(context, enc_ba['par'])
                weighted_coef = enc_ba['num']
                ckks_plain = np.array(ckks_tensor.decrypt(), dtype="float32")
                ckks_plain *= 1.0 / weighted_coef
            else:
                ckks_tensor = ts.ckks_vector_from(context, enc_ba)
                ckks_plain = np.array(ckks_tensor.decrypt(), dtype="float32")
            return ckks_plain
        
        # Collect all cipher items with their indices
        all_cipher_items = []  # [(layer_idx, col_idx, enc_ba), ...]
        for layer_idx, layer_cipher_ba in enumerate(cipher_ba_agg):
            if layer_cipher_ba:
                for col_idx, enc_ba in enumerate(layer_cipher_ba):
                    all_cipher_items.append((layer_idx, col_idx, enc_ba))
        
        # Parallel decryption using thread pool
        decrypted_results = {}  # {layer_idx: [col0, col1, ...]}
        if all_cipher_items:
            with ThreadPoolExecutor(max_workers=min(8, len(all_cipher_items))) as executor:
                futures = {
                    executor.submit(decrypt_single, item[2]): (item[0], item[1])
                    for item in all_cipher_items
                }
                for future in futures:
                    layer_idx, col_idx = futures[future]
                    decrypted = future.result()
                    if layer_idx not in decrypted_results:
                        decrypted_results[layer_idx] = {}
                    decrypted_results[layer_idx][col_idx] = decrypted
        
        # Step 2: Process each layer
        parameters = []
        for layer_idx, layer_cipher_ba in enumerate(cipher_ba_agg):
            ba_plain = np.array(plain_ba_agg[layer_idx])
            out_features, in_features = ba_plain.shape
            selected_cols = enc_a_lines[layer_idx] if layer_idx < len(enc_a_lines) else []

            if not layer_cipher_ba or not selected_cols:
                # No encrypted columns for this layer
                U, s, Vh = np.linalg.svd(ba_plain, full_matrices=False)
                B_agg = U[:, :max_rank] @ np.diag(s[:max_rank])
                A_agg = Vh[:max_rank, :]
                parameters.append(A_agg)
                parameters.append(B_agg)
                continue

            # SVD on plain BA to get B_p, A_p
            U_p, s_p, Vh_p = np.linalg.svd(ba_plain, full_matrices=False)
            r_p = min(max_rank, len(s_p))
            B_p = U_p[:, :r_p] @ np.diag(s_p[:r_p])
            A_p = Vh_p[:r_p, :]

            # Get pre-decrypted columns for this layer
            layer_decrypted = decrypted_results.get(layer_idx, {})
            decrypted_ba_cols = [layer_decrypted[i] for i in sorted(layer_decrypted.keys())]

            if not decrypted_ba_cols:
                parameters.append(A_p[:max_rank, :])
                parameters.append(B_p[:, :max_rank])
                continue

            # Stack decrypted columns: (num_enc_cols, out_features)
            cipher_matrix = np.vstack(decrypted_ba_cols)
            log.debug("Layer %d: cipher_matrix shape %s, B_p shape %s, A_p shape %s",
                      layer_idx, cipher_matrix.shape, B_p.shape, A_p.shape)
            
            # SVD on cipher_matrix.T: (out_features, num_enc_cols)
            # This gives us the cipher part's B_c and A_c
            U_c, s_c, Vh_c = np.linalg.svd(cipher_matrix.T, full_matrices=False)
            # U_c: (out_features, k), s_c: (k,), Vh_c: (k, num_enc_cols)
            r_c = len(s_c)
            B_c = U_c @ np.diag(s_c)  # (out_features, r_c)
            A_c = Vh_c                 # (r_c, num_enc_cols)

            # 3. Merge following Eq. (4) in the paper:
            # B_g = [B_p | B_c] ∈ R^(m×(r_p+r_c))
            # 
            # The key insight: Client-side exchange_columns swapped selected_cols to last positions.
            # After BA computation, A_p's last num_enc cols are zeros (cipher part was zeroed).
            # A_c (r_c, num_enc) contains the cipher part's SVD result.
            #
            # To merge: build A_r = [[A_p, 0], [0, A_c]] of shape (r_p+r_c, in_features+num_enc)
            # Then exchange_columns swaps A_c content into selected_cols positions.
            # Finally take only first in_features columns.
            
            B_g = np.hstack([B_p, B_c])  # (out_features, r_p + r_c)
            
            # Build A_r as block matrix (same as LLM version)
            num_enc = A_c.shape[1]
            A_r = np.block([
                [A_p, np.zeros((A_p.shape[0], num_enc))],
                [np.zeros((A_c.shape[0], in_features)), A_c]
            ])  # (r_p + r_c, in_features + num_enc)
            
            # Exchange columns to restore cipher content to original positions
            A_r = exchange_columns(A_r, selected_cols=selected_cols)
            
            # 4. Final product and SVD to reduce to target rank
            result = B_g @ A_r  # (out_features, in_features + num_enc)
            U_f, s_f, Vh_f = np.linalg.svd(result, full_matrices=False)
            
            # Output: A_agg (max_rank, n), B_agg (out_features, max_rank)
            n_cols = A_p.shape[1]  # in_features
            B_agg = U_f[:, :max_rank] @ np.diag(s_f[:max_rank])
            A_agg = Vh_f[:max_rank, :n_cols]
            
            parameters.append(A_agg)
            parameters.append(B_agg)

        return parameters

    def negotiation_phase(self) -> None:
        """Round 0: Evaluate sensitivity on calibration data and negotiate enc_lines."""
        log.info("=== Negotiation Phase (sensitivity + negotiation) ===")
        device = next(self.clients[0].vision_model.parameters()).device
        lora_r = 8
        if hasattr(self.cfg, "lora_config") and hasattr(self.cfg.lora_config, "r"):
            lora_r = int(self.cfg.lora_config.r)
        self.enc_lines = _negotiation_round(
            self.clients, self.he_budgets, lora_r, device
        )
        log.info("Negotiation done. enc_lines layers (sample): %s", list(self.enc_lines.keys())[:3])

    def run(self) -> str:
        self.setup_clients()
        self.setup_server()
        # Phase 1: Negotiation (sensitivity + determine enc_lines)
        if self.use_she:
            self.negotiation_phase()
        # Phase 2: Federated training rounds
        log.info("=== Training Phase (%d rounds) ===", self.cfg.federated.num_rounds)
        for r in range(self.cfg.federated.num_rounds):
            self.train_round(r)
        # Save final results to results/vision
        out_dir = self._save_results()
        return out_dir or getattr(getattr(self.fabric, "logger", None), "log_dir", ".")

    def _save_results(self) -> str:
        """Save final LoRA state and summary to results/vision/<run_id>/."""
        import json
        from datetime import datetime

        results_root = Path(PROJECT_ROOT) / "results" / "vision"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = results_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA state_dict (from client 0; all clients have same params after aggregation)
        state_dict = get_peft_model_state_dict(self.clients[0].vision_model)
        torch.save({k: v.cpu() for k, v in state_dict.items()}, out_dir / "lora_final.pt")

        # Final evaluation and summary
        summary = {
            "num_rounds": self.cfg.federated.num_rounds,
            "local_steps": self.cfg.federated.local_steps,
            "use_she": self.use_she,
            "clients": [],
        }
        for client in self.clients:
            acc = client.evaluate(client.vision_model, fabric=self.fabric)
            summary["clients"].append({
                "id": client.client_id,
                "dataset": client.dataset_name,
                "test_acc": round(acc, 4),
            })
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        log.info("Results saved to %s", out_dir)
        return str(out_dir)


def get_default_config() -> DictConfig:
    """Default config when no YAML is used (fully self-contained)."""
    return OmegaConf.create({
        "model_name": "clip-vit",
        "dataset_name": "federated",
        "data_root": "./data",
        "model": {
            "model_name_or_path": "openai/clip-vit-base-patch16",
            "input_size": 224,
        },
        "seed": 42,
        "batch_size": 32,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.1,
        "warmup_steps": 100,
        "lora_config": {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        "federated": {
            "num_rounds": 3,
            "local_steps": 50,
            "aggregation_method": "average",
            "evaluate_every_n_rounds": 1,
            "use_she": True,
            "he_budgets": [4, 4],
            "clients": [
                {"id": 0, "dataset": "CIFAR10", "num_samples": 1500},
                {"id": 1, "dataset": "MNIST", "num_samples": 1500},
            ],
        },
    })


def _load_config_with_defaults(config_path: Path, config_file: str) -> DictConfig:
    """Load YAML; if it has a defaults list, load default config first then merge."""
    config_full = config_path / config_file if not os.path.dirname(config_file) else Path(config_file)
    if not config_full.exists():
        return get_default_config()
    file_cfg = OmegaConf.load(config_full)
    defaults = file_cfg.get("defaults", None)
    if defaults and isinstance(defaults, (list, tuple)):
        # Merge default config files first, then overlay current file
        base = get_default_config()
        for name in defaults:
            if isinstance(name, str) and not name.startswith("_"):
                p = config_path / (name if name.endswith(".yaml") else name + ".yaml")
                if p.exists():
                    base = OmegaConf.merge(base, OmegaConf.load(p))
        overlay = {k: v for k, v in file_cfg.items() if k != "defaults"}
        return OmegaConf.merge(base, OmegaConf.create(overlay))
    return OmegaConf.merge(get_default_config(), file_cfg)


def main() -> None:
    _setup_logging()
    config_path = VISION_ROOT / "config"
    # Config file: CLI arg > CONFIG_FILE env > default federated_clip.yaml
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        config_file = sys.argv[1]
        if os.path.dirname(config_file) and not os.path.isabs(config_file):
            config_file = os.path.basename(config_file)
    else:
        config_file = os.environ.get("CONFIG_FILE", "federated_clip.yaml")
    cfg = _load_config_with_defaults(config_path, config_file)
    if "model_name" not in cfg:
        cfg.model_name = "clip-vit"
    if "dataset_name" not in cfg:
        cfg.dataset_name = "federated"
    fabric = setup_fabric(cfg)
    if getattr(fabric, "logger", None) and not os.path.exists(fabric.logger.log_dir):
        os.makedirs(fabric.logger.log_dir)
    if getattr(fabric, "logger", None):
        OmegaConf.save(cfg, os.path.join(fabric.logger.log_dir, "config.yaml"))
    use_she = getattr(cfg.federated, "use_she", True)
    trainer = FederatedCLIPSHETrainer(cfg, fabric, use_she=use_she)
    out = trainer.run()
    log.info("Training done. Output: %s", out)


if __name__ == "__main__":
    main()
