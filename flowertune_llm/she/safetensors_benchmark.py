import csv
import json
import math
import time
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


@dataclass
class BenchmarkMetrics:
    ratio: float
    repeat: int
    client_count: int = 0
    tensor_count: int = 0
    encrypted_tensor_count: int = 0
    encrypted_columns: int = 0
    context_load_seconds: float = 0.0
    load_seconds: float = 0.0
    encrypt_seconds: float = 0.0
    aggregate_plain_seconds: float = 0.0
    aggregate_cipher_seconds: float = 0.0
    decrypt_seconds: float = 0.0
    save_seconds: float = 0.0
    total_seconds: float = 0.0
    baseline_plaintext_bytes: int = 0
    plaintext_upload_bytes: int = 0
    protected_plaintext_bytes: int = 0
    ciphertext_upload_bytes: int = 0
    aggregate_ciphertext_bytes: int = 0
    context_file_bytes: int = 0
    output_file_bytes: int = 0
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0

    def scalar_values(self) -> dict[str, float]:
        values = asdict(self)
        values["model_upload_bytes"] = (
            self.plaintext_upload_bytes + self.ciphertext_upload_bytes
        )
        values["model_roundtrip_bytes"] = (
            values["model_upload_bytes"] + self.aggregate_ciphertext_bytes
        )
        values["aggregate_ciphertext_broadcast_bytes"] = (
            self.aggregate_ciphertext_bytes * self.client_count
        )
        values["model_roundtrip_broadcast_bytes"] = (
            values["model_upload_bytes"]
            + values["aggregate_ciphertext_broadcast_bytes"]
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


def _is_lora_a(key: str, tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 and ".lora_A." in key


def _ratio_label(ratio: float) -> str:
    return f"{ratio:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _load_context(path: Path):
    started = time.perf_counter()
    context = ts.context_from(path.read_bytes())
    return context, time.perf_counter() - started


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


def _mean_lora_a_tensor(
    tensors: list[torch.Tensor], ratio: float, context, metrics: BenchmarkMetrics
) -> tuple[torch.Tensor, list[float]]:
    row_count, column_count = tensors[0].shape
    encrypted_columns = min(column_count, math.ceil(column_count * ratio))
    if encrypted_columns and context is None:
        raise ValueError("A CKKS context is required when encryption_ratio > 0.")
    metrics.encrypted_columns += encrypted_columns
    if encrypted_columns:
        metrics.encrypted_tensor_count += 1

    result = torch.zeros_like(tensors[0], dtype=torch.float64)
    if encrypted_columns < column_count:
        started = time.perf_counter()
        for tensor in tensors:
            result[:, encrypted_columns:].add_(
                tensor[:, encrypted_columns:].to(dtype=torch.float64)
            )
        result[:, encrypted_columns:].div_(len(tensors))
        metrics.aggregate_plain_seconds += time.perf_counter() - started

    errors = []
    for column in range(encrypted_columns):
        aggregate_cipher = None
        expected_sum = np.zeros(row_count, dtype=np.float64)
        for tensor in tensors:
            vector = tensor[:, column].detach().cpu().numpy()
            expected_sum += vector

            started = time.perf_counter()
            serialized = ts.ckks_vector(context, vector).serialize()
            metrics.encrypt_seconds += time.perf_counter() - started
            metrics.ciphertext_upload_bytes += len(serialized)

            started = time.perf_counter()
            incoming = ts.ckks_vector_from(context, serialized)
            if aggregate_cipher is None:
                aggregate_cipher = incoming
            else:
                aggregate_cipher += incoming
            metrics.aggregate_cipher_seconds += time.perf_counter() - started

        started = time.perf_counter()
        aggregate_bytes = aggregate_cipher.serialize()
        metrics.aggregate_cipher_seconds += time.perf_counter() - started
        metrics.aggregate_ciphertext_bytes += len(aggregate_bytes)

        started = time.perf_counter()
        decrypted = np.asarray(
            ts.ckks_vector_from(context, aggregate_bytes).decrypt(), dtype=np.float64
        )
        decrypted /= len(tensors)
        metrics.decrypt_seconds += time.perf_counter() - started

        result[:, column] = torch.from_numpy(decrypted)
        errors.extend(np.abs(decrypted - expected_sum / len(tensors)).tolist())

    element_size = tensors[0].element_size()
    metrics.protected_plaintext_bytes += (
        len(tensors) * row_count * encrypted_columns * element_size
    )
    metrics.plaintext_upload_bytes += (
        len(tensors)
        * row_count
        * (column_count - encrypted_columns)
        * element_size
    )
    return result.to(dtype=tensors[0].dtype), errors


def _write_progress(
    writer: SummaryWriter, metrics: BenchmarkMetrics, tensor_step: int
) -> None:
    global_step = (metrics.repeat - 1) * metrics.tensor_count + tensor_step
    writer.add_scalar(
        "progress/tensors_completed", tensor_step + 1, global_step
    )
    for name in (
        "load_seconds",
        "encrypt_seconds",
        "aggregate_plain_seconds",
        "aggregate_cipher_seconds",
        "decrypt_seconds",
    ):
        writer.add_scalar(
            f"time_cumulative/{name}", getattr(metrics, name), global_step
        )
    writer.add_scalar(
        "network_cumulative/plaintext_upload_bytes",
        metrics.plaintext_upload_bytes,
        global_step,
    )
    writer.add_scalar(
        "network_cumulative/ciphertext_upload_bytes",
        metrics.ciphertext_upload_bytes,
        global_step,
    )
    writer.add_scalar(
        "network_cumulative/aggregate_ciphertext_bytes",
        metrics.aggregate_ciphertext_bytes,
        global_step,
    )


def _write_final_metrics(
    writer: SummaryWriter, metrics: BenchmarkMetrics, step: int
) -> None:
    values = metrics.scalar_values()
    for name in (
        "context_load_seconds",
        "load_seconds",
        "encrypt_seconds",
        "aggregate_plain_seconds",
        "aggregate_cipher_seconds",
        "decrypt_seconds",
        "save_seconds",
        "total_seconds",
    ):
        writer.add_scalar(f"time/{name}", values[name], step)
    for name in (
        "baseline_plaintext_bytes",
        "plaintext_upload_bytes",
        "protected_plaintext_bytes",
        "ciphertext_upload_bytes",
        "aggregate_ciphertext_bytes",
        "aggregate_ciphertext_broadcast_bytes",
        "model_upload_bytes",
        "model_roundtrip_bytes",
        "model_roundtrip_broadcast_bytes",
        "context_file_bytes",
        "output_file_bytes",
        "upload_vs_plain_ratio",
        "ciphertext_expansion_ratio",
    ):
        writer.add_scalar(f"network/{name}", values[name], step)
    for name in (
        "client_count",
        "tensor_count",
        "encrypted_tensor_count",
        "encrypted_columns",
    ):
        writer.add_scalar(f"config/{name}", values[name], step)
    writer.add_scalar("config/encryption_ratio", metrics.ratio, step)
    writer.add_scalar("quality/max_abs_error", metrics.max_abs_error, step)
    writer.add_scalar("quality/mean_abs_error", metrics.mean_abs_error, step)
    writer.flush()


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
    metrics = BenchmarkMetrics(
        ratio=ratio,
        repeat=repeat,
        client_count=len(client_files),
        context_load_seconds=context_load_seconds,
        context_file_bytes=(
            Path(cfg.full_context_path).stat().st_size
            if Path(cfg.full_context_path).is_file()
            else 0
        ),
    )
    output_tensors = {}
    all_errors = []

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
        if cfg.max_tensors is not None:
            keys = keys[: int(cfg.max_tensors)]
        metrics.tensor_count = len(keys)

        writer.add_text("run/tensor_names", "\n".join(keys), repeat - 1)
        for tensor_step, key in enumerate(keys):
            started = time.perf_counter()
            tensors = [handle.get_tensor(key) for handle in handles]
            metrics.load_seconds += time.perf_counter() - started
            _validate_tensor_group(key, tensors)

            tensor_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in tensors
            )
            metrics.baseline_plaintext_bytes += tensor_bytes
            if _is_lora_a(key, tensors[0]):
                output, errors = _mean_lora_a_tensor(
                    tensors, ratio, context, metrics
                )
                output_tensors[key] = output.contiguous()
                all_errors.extend(errors)
            else:
                metrics.plaintext_upload_bytes += tensor_bytes
                output_tensors[key] = _mean_plain_tensor(
                    tensors, metrics
                ).contiguous()
            _write_progress(writer, metrics, tensor_step)

    if all_errors:
        metrics.max_abs_error = max(all_errors)
        metrics.mean_abs_error = sum(all_errors) / len(all_errors)

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "adapter_model.safetensors"
    started = time.perf_counter()
    save_file(
        output_tensors,
        output_path,
        metadata={
            "format": "pt",
            "aggregation": "equal_weight_mean",
            "encryption_ratio": str(ratio),
            "client_count": str(len(client_files)),
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
