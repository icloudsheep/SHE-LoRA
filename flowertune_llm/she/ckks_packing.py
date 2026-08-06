import numpy as np


CKKS_COLUMNS_PER_CIPHERTEXT = 4


def pack_column_blocks(
    columns: np.ndarray,
    columns_per_ciphertext: int = CKKS_COLUMNS_PER_CIPHERTEXT,
) -> list[tuple[np.ndarray, int]]:
    """Pack adjacent matrix columns into column-major CKKS slot vectors."""
    if columns.ndim != 2:
        raise ValueError("CKKS column input must be a matrix.")
    if columns_per_ciphertext < 1:
        raise ValueError("columns_per_ciphertext must be positive.")

    blocks = []
    for start in range(0, columns.shape[1], columns_per_ciphertext):
        block = columns[:, start : start + columns_per_ciphertext]
        blocks.append((block.T.reshape(-1), int(block.shape[1])))
    return blocks


def packed_column_transforms(
    plain_b: np.ndarray,
    column_count: int,
) -> list[np.ndarray]:
    """Build linear transforms that recover each B @ A column from one packed block."""
    if plain_b.ndim != 2:
        raise ValueError("LoRA B input must be a matrix.")
    if column_count < 1:
        raise ValueError("column_count must be positive.")

    output_features, rank = plain_b.shape
    transforms = []
    for column_index in range(column_count):
        transform = np.zeros(
            (rank * column_count, output_features), dtype=np.float64
        )
        start = column_index * rank
        transform[start : start + rank, :] = plain_b.transpose()
        transforms.append(transform)
    return transforms
