import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tenseal as ts
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter

from project_config import load_ckks_benchmark_config
from flowertune_llm.she.ckks_packing import (
    CKKS_COLUMNS_PER_CIPHERTEXT,
    pack_column_blocks,
    packed_column_transforms,
)
from flowertune_llm.she.network_metrics import (
    network_byte_totals,
    serialized_ndarray_size,
)


_WORKER_CONTEXT = None


@dataclass
class BenchmarkMetrics:
    ratio: float
    repeat: int
    client_count: int = 0
    tensor_count: int = 0
    lora_pair_count: int = 0
    encrypted_tensor_count: int = 0
    encrypted_columns: int = 0
    ciphertext_upload_count: int = 0
    ciphertext_columns_per_packet: int = CKKS_COLUMNS_PER_CIPHERTEXT
    client_workers: int = 0
    server_workers: int = 0
    context_load_seconds: float = 0.0
    load_seconds: float = 0.0
    encrypt_seconds: float = 0.0
    encrypt_worker_seconds: float = 0.0
    aggregate_plain_seconds: float = 0.0
    server_multiply_wall_seconds: float = 0.0
    server_multiply_worker_seconds: float = 0.0
    aggregate_cipher_seconds: float = 0.0
    decrypt_seconds: float = 0.0
    validation_seconds: float = 0.0
    reparameterize_seconds: float = 0.0
    save_seconds: float = 0.0
    total_seconds: float = 0.0
    baseline_plaintext_bytes: int = 0
    plaintext_upload_bytes: int = 0
    protected_plaintext_bytes: int = 0
    ciphertext_upload_bytes: int = 0
    plaintext_download_bytes: int = 0
    aggregate_ciphertext_bytes: int = 0
    context_file_bytes: int = 0
    output_file_bytes: int = 0
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0

    def scalar_values(self) -> dict[str, float]:
        values = asdict(self)
        values["aggregation_seconds"] = (
            self.aggregate_plain_seconds
            + self.server_multiply_wall_seconds
            + self.aggregate_cipher_seconds
        )
        values["she_pipeline_seconds"] = (
            self.encrypt_seconds
            + values["aggregation_seconds"]
            + self.decrypt_seconds
        )
        values.update(
            network_byte_totals(
                client_count=self.client_count,
                plaintext_upload_bytes=self.plaintext_upload_bytes,
                ciphertext_upload_bytes=self.ciphertext_upload_bytes,
                plaintext_download_bytes=self.plaintext_download_bytes,
                ciphertext_download_bytes=self.aggregate_ciphertext_bytes,
            )
        )
        values["upload_vs_plain_ratio"] = (
            values["model_upload_bytes"] / self.baseline_plaintext_bytes
            if self.baseline_plaintext_bytes
            else 0.0
        )
        values["ciphertext_expansion_ratio"] = (
            self.ciphertext_upload_bytes / self.protected_plaintext_bytes
            if self.protected_plaintext_bytes
            else 0.0
        )
        return values


def _discover_client_files(cfg) -> list[Path]:
    files = sorted(Path(cfg.input_dir).glob(str(cfg.input_glob)))
    if cfg.max_clients is not None:
        files = files[: int(cfg.max_clients)]
    if not files:
        raise FileNotFoundError(
            f"No client safetensors matched {cfg.input_glob!r} under {cfg.input_dir}."
        )
    return files


def _is_lora_a(key: str) -> bool:
    return ".lora_A." in key


def _is_lora_b(key: str) -> bool:
    return ".lora_B." in key


def _lora_b_key(a_key: str) -> str:
    return a_key.replace(".lora_A.", ".lora_B.", 1)


def _ratio_label(ratio: float) -> str:
    return f"{ratio:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _load_context(path: Path):
    started = time.perf_counter()
    context = ts.context_from(path.read_bytes())
    return context, time.perf_counter() - started


def _init_server_worker(context_path: str) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = ts.context_from(Path(context_path).read_bytes())


def _encrypt_client_columns(
    task: tuple[int, np.ndarray],
) -> tuple[int, list[tuple[bytes, int]], float]:
    client_index, columns = task
    if _WORKER_CONTEXT is None:
        raise RuntimeError("CKKS worker context was not initialized.")

    started = time.perf_counter()
    encrypted_blocks = [
        (
            ts.ckks_vector(_WORKER_CONTEXT, packed_values).serialize(),
            column_count,
        )
        for packed_values, column_count in pack_column_blocks(columns)
    ]
    return client_index, encrypted_blocks, time.perf_counter() - started


def _multiply_client_cipher(
    task: tuple[int, np.ndarray, list[tuple[bytes, int]]],
) -> tuple[int, list[bytes], float]:
    client_index, scaled_b, encrypted_blocks = task
    if _WORKER_CONTEXT is None:
        raise RuntimeError("CKKS worker context was not initialized.")

    started = time.perf_counter()
    results = []
    transforms_by_width = {}
    for encrypted_block, column_count in encrypted_blocks:
        cipher_a = ts.ckks_vector_from(_WORKER_CONTEXT, encrypted_block)
        transforms = transforms_by_width.get(column_count)
        if transforms is None:
            transforms = [
                ts.plain_tensor(transform)
                for transform in packed_column_transforms(scaled_b, column_count)
            ]
            transforms_by_width[column_count] = transforms
        results.extend(cipher_a.mm(transform).serialize() for transform in transforms)
    return client_index, results, time.perf_counter() - started


def _validate_tensor_group(
    key: str, tensors: list[torch.Tensor]
) -> tuple[torch.Size, torch.dtype]:
    shape = tensors[0].shape
    dtype = tensors[0].dtype
    for client_index, tensor in enumerate(tensors[1:], start=1):
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(
                f"Tensor {key!r} differs for client index {client_index}: "
                f"expected shape={tuple(shape)}, dtype={dtype}; "
                f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}."
            )
    if not tensors[0].is_floating_point():
        raise TypeError(f"Tensor {key!r} must use a floating-point dtype.")
    return shape, dtype


def _validate_lora_pair(
    a_key: str, a_tensors: list[torch.Tensor], b_tensors: list[torch.Tensor]
) -> tuple[int, int, int, torch.dtype]:
    a_shape, dtype = _validate_tensor_group(a_key, a_tensors)
    b_key = _lora_b_key(a_key)
    b_shape, b_dtype = _validate_tensor_group(b_key, b_tensors)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"LoRA pair {a_key!r} and {b_key!r} must be matrices.")
    rank, input_features = a_shape
    output_features, b_rank = b_shape
    if rank != b_rank:
        raise ValueError(
            f"LoRA pair rank mismatch for {a_key!r}: A={tuple(a_shape)}, "
            f"B={tuple(b_shape)}."
        )
    if dtype != b_dtype:
        raise ValueError(
            f"LoRA pair dtype mismatch for {a_key!r}: A={dtype}, B={b_dtype}."
        )
    return int(rank), int(input_features), int(output_features), dtype


def _mean_plain_tensor(
    tensors: list[torch.Tensor], metrics: BenchmarkMetrics
) -> torch.Tensor:
    started = time.perf_counter()
    result = torch.zeros_like(tensors[0], dtype=torch.float64)
    for tensor in tensors:
        result.add_(tensor.to(dtype=torch.float64))
    result.div_(len(tensors))
    metrics.aggregate_plain_seconds += time.perf_counter() - started
    return result.to(dtype=tensors[0].dtype)


def _compress_ba(
    ba_matrix: np.ndarray, rank: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    truncated_rank = min(rank, ba_matrix.shape[0], ba_matrix.shape[1])
    if truncated_rank < 1:
        raise ValueError("LoRA rank must be positive.")
    u, singular_values, vh = np.linalg.svd(ba_matrix, full_matrices=False)
    aggregated_b = u[:, :truncated_rank] * singular_values[:truncated_rank]
    aggregated_a = vh[:truncated_rank, :]
    return (
        torch.from_numpy(aggregated_a.copy()).to(dtype=dtype),
        torch.from_numpy(aggregated_b.copy()).to(dtype=dtype),
    )


def _aggregate_cipher_batch(
    client_results: list[list[bytes]],
    context,
    metrics: BenchmarkMetrics,
) -> list[np.ndarray]:
    if not client_results:
        return []
    batch_size = len(client_results[0])
    if any(len(result) != batch_size for result in client_results):
        raise ValueError("Clients returned different encrypted column counts.")

    decrypted_columns = []
    for column_index in range(batch_size):
        started = time.perf_counter()
        aggregate_cipher = None
        for result in client_results:
            incoming = ts.ckks_vector_from(context, result[column_index])
            if aggregate_cipher is None:
                aggregate_cipher = incoming
            else:
                aggregate_cipher += incoming
        aggregate_bytes = aggregate_cipher.serialize()
        metrics.aggregate_cipher_seconds += time.perf_counter() - started
        metrics.aggregate_ciphertext_bytes += len(aggregate_bytes)

        started = time.perf_counter()
        decrypted = np.asarray(
            ts.ckks_vector_from(context, aggregate_bytes).decrypt(), dtype=np.float64
        )
        metrics.decrypt_seconds += time.perf_counter() - started
        decrypted_columns.append(decrypted)
    return decrypted_columns


def _aggregate_lora_pair(
    a_key: str,
    a_tensors: list[torch.Tensor],
    b_tensors: list[torch.Tensor],
    ratio: float,
    context,
    client_pool: ProcessPoolExecutor | None,
    server_pool: ProcessPoolExecutor | None,
    batch_columns: int,
    metrics: BenchmarkMetrics,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    rank, input_features, output_features, dtype = _validate_lora_pair(
        a_key, a_tensors, b_tensors
    )
    encrypted_count = min(input_features, math.floor(input_features * ratio))
    client_count = len(a_tensors)
    client_scale = 1.0 / client_count

    metrics.encrypted_columns += encrypted_count
    if encrypted_count:
        metrics.encrypted_tensor_count += 1

    a_arrays = [tensor.detach().cpu().numpy().astype(np.float64) for tensor in a_tensors]
    b_arrays = [tensor.detach().cpu().numpy().astype(np.float64) for tensor in b_tensors]

    started = time.perf_counter()
    plain_ba = np.zeros((output_features, input_features), dtype=np.float64)
    for a_matrix, b_matrix in zip(a_arrays, b_arrays):
        plain_a = a_matrix.copy()
        plain_a[:, input_features - encrypted_count :] = 0.0
        plain_ba += (b_matrix * client_scale) @ plain_a
    metrics.aggregate_plain_seconds += time.perf_counter() - started

    error_max = 0.0
    error_sum = 0.0
    error_count = 0
    if encrypted_count:
        if context is None or client_pool is None or server_pool is None:
            raise ValueError(
                "CKKS context and client/server process pools are required."
            )
        decrypted_ba = np.empty((output_features, encrypted_count), dtype=np.float64)

        protected_start = input_features - encrypted_count
        for batch_start in range(0, encrypted_count, batch_columns):
            batch_end = min(batch_start + batch_columns, encrypted_count)
            source_start = protected_start + batch_start
            source_end = protected_start + batch_end
            expected = np.zeros(
                (output_features, batch_end - batch_start), dtype=np.float64
            )

            for a_matrix, b_matrix in zip(a_arrays, b_arrays):
                started = time.perf_counter()
                expected += (b_matrix * client_scale) @ a_matrix[
                    :, source_start:source_end
                ]
                metrics.validation_seconds += time.perf_counter() - started

            encryption_tasks = [
                (client_index, a_matrix[:, source_start:source_end])
                for client_index, a_matrix in enumerate(a_arrays)
            ]
            started = time.perf_counter()
            futures = [
                client_pool.submit(_encrypt_client_columns, task)
                for task in encryption_tasks
            ]
            encrypted_by_client = [None] * client_count
            for future in futures:
                client_index, encrypted_columns, worker_seconds = future.result()
                encrypted_by_client[client_index] = encrypted_columns
                metrics.encrypt_worker_seconds += worker_seconds
                metrics.ciphertext_upload_count += len(encrypted_columns)
                metrics.ciphertext_upload_bytes += sum(
                    len(ciphertext) for ciphertext, _ in encrypted_columns
                )
            metrics.encrypt_seconds += time.perf_counter() - started

            tasks = [
                (client_index, b_arrays[client_index] * client_scale, client_cipher)
                for client_index, client_cipher in enumerate(encrypted_by_client)
            ]
            started = time.perf_counter()
            futures = [
                server_pool.submit(_multiply_client_cipher, task) for task in tasks
            ]
            ordered_results = [None] * client_count
            for future in futures:
                client_index, result, worker_seconds = future.result()
                ordered_results[client_index] = result
                metrics.server_multiply_worker_seconds += worker_seconds
            metrics.server_multiply_wall_seconds += time.perf_counter() - started

            decrypted_columns = _aggregate_cipher_batch(
                ordered_results, context, metrics
            )
            decrypted_batch = np.column_stack(decrypted_columns)
            decrypted_ba[:, batch_start:batch_end] = decrypted_batch
            absolute_error = np.abs(decrypted_batch - expected)
            error_max = max(error_max, float(absolute_error.max(initial=0.0)))
            error_sum += float(absolute_error.sum())
            error_count += int(absolute_error.size)

        plain_ba[:, protected_start:] = decrypted_ba

    started = time.perf_counter()
    aggregated_a, aggregated_b = _compress_ba(plain_ba, rank, dtype)
    metrics.reparameterize_seconds += time.perf_counter() - started
    return aggregated_a, aggregated_b, error_max, error_sum, error_count


def _write_progress(
    writer: SummaryWriter, metrics: BenchmarkMetrics, pair_step: int
) -> None:
    global_step = (metrics.repeat - 1) * metrics.lora_pair_count + pair_step
    writer.add_scalar("progress/lora_pairs_completed", pair_step + 1, global_step)
    for name in (
        "load_seconds",
        "encrypt_seconds",
        "encrypt_worker_seconds",
        "aggregate_plain_seconds",
        "server_multiply_wall_seconds",
        "server_multiply_worker_seconds",
        "aggregate_cipher_seconds",
        "decrypt_seconds",
        "validation_seconds",
        "reparameterize_seconds",
    ):
        writer.add_scalar(
            f"time_cumulative/{name}", getattr(metrics, name), global_step
        )
    for name in (
        "plaintext_upload_bytes",
        "ciphertext_upload_bytes",
        "plaintext_download_bytes",
        "aggregate_ciphertext_bytes",
        "ciphertext_upload_count",
    ):
        writer.add_scalar(
            f"network_cumulative/{name}", getattr(metrics, name), global_step
        )


def _write_final_metrics(
    writer: SummaryWriter, metrics: BenchmarkMetrics, step: int
) -> None:
    values = metrics.scalar_values()
    for name in (
        "context_load_seconds",
        "load_seconds",
        "encrypt_seconds",
        "encrypt_worker_seconds",
        "aggregate_plain_seconds",
        "server_multiply_wall_seconds",
        "server_multiply_worker_seconds",
        "aggregate_cipher_seconds",
        "aggregation_seconds",
        "decrypt_seconds",
        "she_pipeline_seconds",
        "validation_seconds",
        "reparameterize_seconds",
        "save_seconds",
        "total_seconds",
    ):
        writer.add_scalar(f"time/{name}", values[name], step)
    for name in (
        "baseline_plaintext_bytes",
        "plaintext_upload_bytes",
        "protected_plaintext_bytes",
        "ciphertext_upload_bytes",
        "plaintext_download_bytes",
        "aggregate_ciphertext_bytes",
        "plaintext_download_broadcast_bytes",
        "aggregate_ciphertext_broadcast_bytes",
        "model_upload_bytes",
        "model_upload_per_client_bytes",
        "model_download_bytes",
        "model_download_per_client_bytes",
        "model_download_broadcast_bytes",
        "model_roundtrip_bytes",
        "model_roundtrip_broadcast_bytes",
        "context_file_bytes",
        "output_file_bytes",
        "upload_vs_plain_ratio",
        "ciphertext_expansion_ratio",
        "ciphertext_upload_count",
    ):
        writer.add_scalar(f"network/{name}", values[name], step)
    for name in (
        "client_count",
        "tensor_count",
        "lora_pair_count",
        "encrypted_tensor_count",
        "encrypted_columns",
        "ciphertext_columns_per_packet",
        "client_workers",
        "server_workers",
    ):
        writer.add_scalar(f"config/{name}", values[name], step)
    writer.add_scalar("config/encryption_ratio", metrics.ratio, step)
    writer.add_scalar("quality/max_abs_error", metrics.max_abs_error, step)
    writer.add_scalar("quality/mean_abs_error", metrics.mean_abs_error, step)
    writer.flush()


def _select_keys(keys: list[str], max_pairs: int | None) -> tuple[list[str], list[str]]:
    a_keys = [key for key in keys if _is_lora_a(key)]
    b_keys = {key for key in keys if _is_lora_b(key)}
    if max_pairs is not None:
        a_keys = a_keys[:max_pairs]
    if not a_keys:
        raise ValueError("No LoRA A/B tensor pairs were found in the client files.")
    for a_key in a_keys:
        expected_b = _lora_b_key(a_key)
        if expected_b not in b_keys:
            raise ValueError(f"Missing LoRA B tensor for {a_key!r}: {expected_b!r}.")
    selected_lora = set(a_keys) | {_lora_b_key(key) for key in a_keys}
    other_keys = (
        []
        if max_pairs is not None
        else [key for key in keys if key not in selected_lora]
    )
    unmatched_b = [
        key
        for key in other_keys
        if _is_lora_b(key)
        and key.replace(".lora_B.", ".lora_A.", 1) not in set(a_keys)
    ]
    if unmatched_b:
        raise ValueError(f"LoRA B tensor has no matching A tensor: {unmatched_b[0]!r}.")
    return a_keys, other_keys


def _run_repeat(
    cfg,
    client_files: list[Path],
    ratio: float,
    repeat: int,
    context,
    context_load_seconds: float,
    output_dir: Path,
    writer: SummaryWriter,
) -> BenchmarkMetrics:
    total_started = time.perf_counter()
    requested_client_workers = (
        int(cfg.client_workers)
        if cfg.client_workers is not None
        else (os.cpu_count() or 1)
    )
    requested_server_workers = (
        int(cfg.server_workers)
        if cfg.server_workers is not None
        else (os.cpu_count() or 1)
    )
    effective_client_workers = (
        min(len(client_files), requested_client_workers) if ratio > 0 else 0
    )
    effective_server_workers = (
        min(len(client_files), requested_server_workers) if ratio > 0 else 0
    )
    metrics = BenchmarkMetrics(
        ratio=ratio,
        repeat=repeat,
        client_count=len(client_files),
        client_workers=effective_client_workers,
        server_workers=effective_server_workers,
        context_load_seconds=context_load_seconds,
        context_file_bytes=(
            Path(cfg.full_context_path).stat().st_size
            if Path(cfg.full_context_path).is_file()
            else 0
        ),
    )
    output_tensors = {}
    error_max = 0.0
    error_sum = 0.0
    error_count = 0

    client_pool = None
    server_pool = None
    if ratio > 0:
        client_pool = ProcessPoolExecutor(
            max_workers=effective_client_workers,
            initializer=_init_server_worker,
            initargs=(str(cfg.full_context_path),),
        )
        server_pool = ProcessPoolExecutor(
            max_workers=effective_server_workers,
            initializer=_init_server_worker,
            initargs=(str(cfg.full_context_path),),
        )

    try:
        with ExitStack() as stack:
            started = time.perf_counter()
            handles = [
                stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                for path in client_files
            ]
            metrics.load_seconds += time.perf_counter() - started
            keys = list(handles[0].keys())
            for client_index, handle in enumerate(handles[1:], start=1):
                if list(handle.keys()) != keys:
                    raise ValueError(
                        f"Client index {client_index} has different tensor names."
                    )

            a_keys, other_keys = _select_keys(keys, cfg.max_tensors)
            metrics.lora_pair_count = len(a_keys)
            metrics.tensor_count = len(a_keys) * 2 + len(other_keys)
            writer.add_text("run/lora_a_tensor_names", "\n".join(a_keys), repeat - 1)

            for pair_step, a_key in enumerate(a_keys):
                b_key = _lora_b_key(a_key)
                started = time.perf_counter()
                a_tensors = [handle.get_tensor(a_key) for handle in handles]
                b_tensors = [handle.get_tensor(b_key) for handle in handles]
                metrics.load_seconds += time.perf_counter() - started

                pair_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in a_tensors + b_tensors
                )
                metrics.baseline_plaintext_bytes += pair_bytes
                metrics.plaintext_upload_bytes += sum(
                    serialized_ndarray_size(tensor.detach().cpu().numpy())
                    for tensor in a_tensors + b_tensors
                )
                encrypted_count = min(
                    a_tensors[0].shape[1],
                    math.floor(a_tensors[0].shape[1] * ratio),
                )
                metrics.protected_plaintext_bytes += sum(
                    tensor.shape[0] * encrypted_count * tensor.element_size()
                    for tensor in a_tensors
                )

                (
                    aggregated_a,
                    aggregated_b,
                    pair_error_max,
                    pair_error_sum,
                    pair_error_count,
                ) = _aggregate_lora_pair(
                    a_key,
                    a_tensors,
                    b_tensors,
                    ratio,
                    context,
                    client_pool,
                    server_pool,
                    int(cfg.ciphertext_batch_columns),
                    metrics,
                )
                output_tensors[a_key] = aggregated_a.contiguous()
                output_tensors[b_key] = aggregated_b.contiguous()
                metrics.plaintext_download_bytes += sum(
                    serialized_ndarray_size(tensor.detach().cpu().numpy())
                    for tensor in (aggregated_a, aggregated_b)
                )
                error_max = max(error_max, pair_error_max)
                error_sum += pair_error_sum
                error_count += pair_error_count
                _write_progress(writer, metrics, pair_step)

            for key in other_keys:
                started = time.perf_counter()
                tensors = [handle.get_tensor(key) for handle in handles]
                metrics.load_seconds += time.perf_counter() - started
                _validate_tensor_group(key, tensors)
                tensor_bytes = sum(
                    tensor.numel() * tensor.element_size() for tensor in tensors
                )
                metrics.baseline_plaintext_bytes += tensor_bytes
                metrics.plaintext_upload_bytes += sum(
                    serialized_ndarray_size(tensor.detach().cpu().numpy())
                    for tensor in tensors
                )
                aggregated_tensor = _mean_plain_tensor(tensors, metrics).contiguous()
                output_tensors[key] = aggregated_tensor
                metrics.plaintext_download_bytes += serialized_ndarray_size(
                    aggregated_tensor.detach().cpu().numpy()
                )
    finally:
        if client_pool is not None:
            client_pool.shutdown(wait=True)
        if server_pool is not None:
            server_pool.shutdown(wait=True)

    if error_count:
        metrics.max_abs_error = error_max
        metrics.mean_abs_error = error_sum / error_count

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "adapter_model.safetensors"
    started = time.perf_counter()
    save_file(
        output_tensors,
        output_path,
        metadata={
            "format": "pt",
            "aggregation": "equal_weight_mean_ba_then_rank_compression",
            "encryption_ratio": str(ratio),
            "client_count": str(len(client_files)),
            "client_workers": str(effective_client_workers),
            "server_workers": str(effective_server_workers),
        },
    )
    metrics.save_seconds = time.perf_counter() - started
    metrics.output_file_bytes = output_path.stat().st_size
    metrics.total_seconds = time.perf_counter() - total_started
    _write_final_metrics(writer, metrics, repeat - 1)
    return metrics


def _write_summary(run_dir: Path, rows: list[BenchmarkMetrics]) -> None:
    payload = [row.scalar_values() for row in rows]
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)


def run_benchmark() -> Path:
    cfg = load_ckks_benchmark_config(validate_paths=True)
    client_files = _discover_client_files(cfg)
    run_name = str(cfg.run_name).strip() or datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run_dir = Path(cfg.output_dir) / run_name
    tensorboard_run_dir = Path(cfg.tensorboard_dir) / run_name
    if run_dir.exists() or tensorboard_run_dir.exists():
        raise FileExistsError(
            f"Benchmark run already exists in output or TensorBoard logs: {run_name}"
        )
    run_dir.mkdir(parents=True)

    if any(float(ratio) > 0.0 for ratio in cfg.encryption_ratios):
        context, context_load_seconds = _load_context(Path(cfg.full_context_path))
    else:
        context, context_load_seconds = None, 0.0
    rows = []
    try:
        for ratio in cfg.encryption_ratios:
            ratio_label = _ratio_label(float(ratio))
            writer = SummaryWriter(
                log_dir=str(tensorboard_run_dir / f"ratio_{ratio_label}")
            )
            try:
                writer.add_text("run/config", OmegaConf.to_yaml(cfg), 0)
                writer.add_text(
                    "run/client_files",
                    "\n".join(str(path) for path in client_files),
                    0,
                )
                writer.add_text(
                    "run/execution_topology",
                    (
                        "client encryption: process-parallel by client and serial by column; "
                        "server B*Enc(A): process-parallel by client; "
                        "ciphertext aggregation: serial; decryption: serial"
                    ),
                    0,
                )
                for repeat in range(1, int(cfg.repeats) + 1):
                    repeat_dir = (
                        run_dir
                        / f"ratio_{ratio_label}"
                        / f"repeat_{repeat:03d}"
                    )
                    metrics = _run_repeat(
                        cfg,
                        client_files,
                        float(ratio),
                        repeat,
                        context,
                        context_load_seconds,
                        repeat_dir,
                        writer,
                    )
                    rows.append(metrics)
                    print(
                        f"ratio={float(ratio):.2%} repeat={repeat} "
                        f"client_workers={metrics.client_workers} "
                        f"server_workers={metrics.server_workers} "
                        f"total={metrics.total_seconds:.3f}s "
                        f"upload={metrics.scalar_values()['model_upload_bytes']} bytes"
                    )
            except Exception as error:
                writer.add_text(
                    "run/error",
                    f"{type(error).__name__}: {error}",
                    max(0, len(rows)),
                )
                writer.flush()
                raise
            finally:
                writer.flush()
                writer.close()
    finally:
        if rows:
            _write_summary(run_dir, rows)
    print(f"Results: {run_dir}")
    print(f"TensorBoard: {tensorboard_run_dir}")
    return run_dir


if __name__ == "__main__":
    run_benchmark()
