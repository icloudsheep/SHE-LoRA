from io import BytesIO

import numpy as np


def serialized_ndarray_size(array: np.ndarray) -> int:
    """Return the Flower tensor payload size for one NumPy array."""
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.tell()


def network_byte_totals(
    *,
    client_count: int,
    plaintext_upload_bytes: int,
    ciphertext_upload_bytes: int,
    plaintext_download_bytes: int,
    ciphertext_download_bytes: int,
) -> dict[str, float]:
    """Calculate per-client and all-client traffic for one federated round."""
    if client_count < 1:
        raise ValueError("client_count must be positive.")

    upload_all_clients = plaintext_upload_bytes + ciphertext_upload_bytes
    upload_per_client = upload_all_clients / client_count
    download_per_client = plaintext_download_bytes + ciphertext_download_bytes
    plaintext_download_broadcast = plaintext_download_bytes * client_count
    ciphertext_download_broadcast = ciphertext_download_bytes * client_count
    download_broadcast = download_per_client * client_count

    return {
        "model_upload_bytes": upload_all_clients,
        "model_upload_per_client_bytes": upload_per_client,
        "model_download_bytes": download_per_client,
        "model_download_per_client_bytes": download_per_client,
        "model_roundtrip_bytes": upload_per_client + download_per_client,
        "plaintext_download_broadcast_bytes": plaintext_download_broadcast,
        "aggregate_ciphertext_broadcast_bytes": ciphertext_download_broadcast,
        "model_download_broadcast_bytes": download_broadcast,
        "model_roundtrip_broadcast_bytes": upload_all_clients + download_broadcast,
    }
