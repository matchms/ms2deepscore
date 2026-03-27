from typing import Tuple
import numpy as np
from numba import jit, prange
from chemap.metrics import (
    tanimoto_similarity_dense,
    tanimoto_similarity_sparse,
    tanimoto_similarity_sparse_binary,
    tanimoto_similarity_matrix_dense,
    tanimoto_similarity_matrix_sparse_binary,
    tanimoto_similarity_matrix_sparse,
)


DENSE_FINGERPRINT_TYPES = {"rdkit_binary", "rdkit_count", "rdkit_logcount"}
UNFOLDED_BINARY_FINGERPRINT_TYPES = {"rdkit_binary_unfolded"}
UNFOLDED_COUNT_FINGERPRINT_TYPES = {"rdkit_count_unfolded", "rdkit_logcount_unfolded"}
SUPPORTED_FINGERPRINT_TYPES = (
    DENSE_FINGERPRINT_TYPES
    | UNFOLDED_BINARY_FINGERPRINT_TYPES
    | UNFOLDED_COUNT_FINGERPRINT_TYPES
)


def is_dense_fingerprint_type(fingerprint_type: str) -> bool:
    return fingerprint_type in DENSE_FINGERPRINT_TYPES


def is_unfolded_binary_fingerprint_type(fingerprint_type: str) -> bool:
    return fingerprint_type in UNFOLDED_BINARY_FINGERPRINT_TYPES


def is_unfolded_count_fingerprint_type(fingerprint_type: str) -> bool:
    return fingerprint_type in UNFOLDED_COUNT_FINGERPRINT_TYPES


def compute_fingerprint_similarity_matrix(
    fingerprints_1,
    fingerprints_2,
    fingerprint_type: str,
) -> np.ndarray:
    """Compute pairwise Tanimoto similarities for any supported fingerprint type."""
    if fingerprint_type not in SUPPORTED_FINGERPRINT_TYPES:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")

    if is_dense_fingerprint_type(fingerprint_type):
        return tanimoto_similarity_matrix_dense(fingerprints_1, fingerprints_2)

    if is_unfolded_binary_fingerprint_type(fingerprint_type):
        return tanimoto_similarity_matrix_sparse_binary(fingerprints_1, fingerprints_2)

    if is_unfolded_count_fingerprint_type(fingerprint_type):
        return tanimoto_similarity_matrix_sparse(
            [x[0] for x in fingerprints_1],
            [x[1] for x in fingerprints_1],
            [x[0] for x in fingerprints_2],
            [x[1] for x in fingerprints_2],
        )

    raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")


def compute_fingerprint_similarity_row(
    single_fingerprint,
    fingerprints,
    fingerprint_type: str,
) -> np.ndarray:
    """Compute similarities of one fingerprint to a collection of fingerprints."""
    if fingerprint_type not in SUPPORTED_FINGERPRINT_TYPES:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")

    if is_dense_fingerprint_type(fingerprint_type):
        size = fingerprints.shape[0]
        tanimoto_scores = np.zeros(size, dtype=np.float32)
        for idx_fingerprint_j in range(size):
            fingerprint_j = fingerprints[idx_fingerprint_j, :]
            tanimoto_scores[idx_fingerprint_j] = tanimoto_similarity_dense(
                single_fingerprint, fingerprint_j
            )
        return tanimoto_scores

    # For unfolded fingerprints, use the matrix function and take row 0.
    return compute_fingerprint_similarity_matrix(
        [single_fingerprint], fingerprints, fingerprint_type
    )[0]


# Add row based similarity computations
# -------------------------------------


@jit(nopython=True, fastmath=True)
def tanimoto_scores_row_dense(single_fingerprint, list_of_fingerprints):
    size = list_of_fingerprints.shape[0]
    tanimoto_scores = np.zeros(size)

    for idx_fingerprint_j in range(size):
        fingerprint_j = list_of_fingerprints[idx_fingerprint_j, :]
        tanimoto_score = tanimoto_similarity_dense(single_fingerprint, fingerprint_j)
        tanimoto_scores[idx_fingerprint_j] = tanimoto_score
    return tanimoto_scores


@jit(nopython=True, fastmath=True)
def tanimoto_scores_row_sparse_binary(single_fingerprint, list_of_fingerprints):
    size = len(list_of_fingerprints)
    tanimoto_scores = np.zeros(size, dtype=np.float32)

    for idx_fingerprint_j in range(size):
        fingerprint_j = list_of_fingerprints[idx_fingerprint_j]
        tanimoto_scores[idx_fingerprint_j] = tanimoto_similarity_sparse_binary(
            single_fingerprint, fingerprint_j
        )
    return tanimoto_scores


@jit(nopython=True, fastmath=True)
def tanimoto_scores_row_sparse_count(
    single_bins,
    single_counts,
    list_of_bins,
    list_of_counts,
):
    size = len(list_of_bins)
    tanimoto_scores = np.zeros(size, dtype=np.float32)

    for idx_fingerprint_j in range(size):
        bins_j = list_of_bins[idx_fingerprint_j]
        counts_j = list_of_counts[idx_fingerprint_j]
        tanimoto_scores[idx_fingerprint_j] = tanimoto_similarity_sparse(
            single_bins, single_counts, bins_j, counts_j
        )
    return tanimoto_scores


from numba import jit
import numpy as np


@jit(nopython=True, fastmath=True)
def _fill_pairs_for_row_same_set(
    selected_pairs_per_bin,
    selected_scores_per_bin,
    tanimoto_scores,
    idx_fingerprint_i,
    max_pairs_per_bin,
    selection_bins,
    include_diagonal,
):
    num_bins = len(selection_bins)

    for bin_number in range(num_bins):
        selection_bin = selection_bins[bin_number]
        indices = np.nonzero(
            (tanimoto_scores > selection_bin[0]) & (tanimoto_scores <= selection_bin[1])
        )[0]

        if not include_diagonal and idx_fingerprint_i in indices:
            indices = indices[indices != idx_fingerprint_i]

        np.random.shuffle(indices)
        indices = indices[:max_pairs_per_bin]
        num_indices = len(indices)

        selected_pairs_per_bin[bin_number, idx_fingerprint_i, :num_indices] = indices
        selected_scores_per_bin[bin_number, idx_fingerprint_i, :num_indices] = tanimoto_scores[indices]


@jit(nopython=True, parallel=True)
def _compute_tanimoto_similarity_per_bin_dense(
    fingerprints,
    max_pairs_per_bin,
    selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)], dtype=np.float32),
    include_diagonal=True,
) -> Tuple[np.ndarray, np.ndarray]:
    size = fingerprints.shape[0]
    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size, max_pairs_per_bin), dtype=np.float32)

    for idx_fingerprint_i in prange(size):
        fingerprint_i = fingerprints[idx_fingerprint_i, :]
        tanimoto_scores = tanimoto_scores_row_dense(fingerprint_i, fingerprints)

        _fill_pairs_for_row_same_set(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            idx_fingerprint_i,
            max_pairs_per_bin,
            selection_bins,
            include_diagonal,
        )

    return selected_pairs_per_bin, selected_scores_per_bin


@jit(nopython=True, parallel=True)
def _compute_tanimoto_similarity_per_bin_sparse_binary(
    fingerprints,
    max_pairs_per_bin,
    selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)], dtype=np.float32),
    include_diagonal=True,
) -> Tuple[np.ndarray, np.ndarray]:
    size = len(fingerprints)
    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size, max_pairs_per_bin), dtype=np.float32)

    for idx_fingerprint_i in prange(size):
        fingerprint_i = fingerprints[idx_fingerprint_i]
        tanimoto_scores = tanimoto_scores_row_sparse_binary(fingerprint_i, fingerprints)

        _fill_pairs_for_row_same_set(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            idx_fingerprint_i,
            max_pairs_per_bin,
            selection_bins,
            include_diagonal,
        )

    return selected_pairs_per_bin, selected_scores_per_bin


@jit(nopython=True, parallel=True)
def _compute_tanimoto_similarity_per_bin_sparse_count(
    fingerprints_bins,
    fingerprints_counts,
    max_pairs_per_bin,
    selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)], dtype=np.float32),
    include_diagonal=True,
) -> Tuple[np.ndarray, np.ndarray]:
    size = len(fingerprints_bins)
    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size, max_pairs_per_bin), dtype=np.float32)

    for idx_fingerprint_i in prange(size):
        bins_i = fingerprints_bins[idx_fingerprint_i]
        counts_i = fingerprints_counts[idx_fingerprint_i]
        tanimoto_scores = tanimoto_scores_row_sparse_count(
            bins_i, counts_i, fingerprints_bins, fingerprints_counts
        )

        _fill_pairs_for_row_same_set(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            idx_fingerprint_i,
            max_pairs_per_bin,
            selection_bins,
            include_diagonal,
        )

    return selected_pairs_per_bin, selected_scores_per_bin


def _split_sparse_count_fingerprints(fingerprints):
    fingerprint_bins = [x[0] for x in fingerprints]
    fingerprint_counts = [x[1] for x in fingerprints]
    return fingerprint_bins, fingerprint_counts


def compute_tanimoto_similarity_per_bin(
    fingerprints,
    max_pairs_per_bin,
    fingerprint_type: str,
    selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)], dtype=np.float32),
    include_diagonal=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate pairwise-per-bin Tanimoto implementation."""
    if fingerprint_type not in SUPPORTED_FINGERPRINT_TYPES:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")

    if is_dense_fingerprint_type(fingerprint_type):
        return _compute_tanimoto_similarity_per_bin_dense(
            fingerprints,
            max_pairs_per_bin=max_pairs_per_bin,
            selection_bins=selection_bins,
            include_diagonal=include_diagonal,
        )

    if is_unfolded_binary_fingerprint_type(fingerprint_type):
        return _compute_tanimoto_similarity_per_bin_sparse_binary(
            fingerprints,
            max_pairs_per_bin=max_pairs_per_bin,
            selection_bins=selection_bins,
            include_diagonal=include_diagonal,
        )

    if is_unfolded_count_fingerprint_type(fingerprint_type):
        fingerprint_bins, fingerprint_counts = _split_sparse_count_fingerprints(fingerprints)
        return _compute_tanimoto_similarity_per_bin_sparse_count(
            fingerprint_bins,
            fingerprint_counts,
            max_pairs_per_bin=max_pairs_per_bin,
            selection_bins=selection_bins,
            include_diagonal=include_diagonal,
        )

    raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")


@jit(nopython=True, fastmath=True)
def _fill_pairs_for_row_between_sets(
    selected_pairs_per_bin,
    selected_scores_per_bin,
    tanimoto_scores,
    row_index,
    target_offset,
    max_pairs_per_bin,
    selection_bins,
):
    num_bins = len(selection_bins)

    for bin_number in range(num_bins):
        selection_bin = selection_bins[bin_number]
        indices = np.nonzero(
            (tanimoto_scores > selection_bin[0]) & (tanimoto_scores <= selection_bin[1])
        )[0]

        np.random.shuffle(indices)
        indices = indices[:max_pairs_per_bin]
        num_indices = len(indices)

        selected_pairs_per_bin[bin_number, row_index, :num_indices] = indices + target_offset
        selected_scores_per_bin[bin_number, row_index, :num_indices] = tanimoto_scores[indices]


@jit(nopython=True, parallel=True)
def _compute_tanimoto_similarity_per_bin_between_sets_dense(
    fingerprints_1,
    fingerprints_2,
    max_pairs_per_bin,
    selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)], dtype=np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    size_1 = fingerprints_1.shape[0]
    size_2 = fingerprints_2.shape[0]
    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size_1 + size_2, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size_1 + size_2, max_pairs_per_bin), dtype=np.float32)

    for idx_fingerprint_i in prange(size_1):
        fingerprint_i = fingerprints_1[idx_fingerprint_i, :]
        tanimoto_scores = tanimoto_scores_row_dense(fingerprint_i, fingerprints_2)

        _fill_pairs_for_row_between_sets(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            idx_fingerprint_i,
            size_1,
            max_pairs_per_bin,
            selection_bins,
        )

    for idx_fingerprint_j in prange(size_2):
        fingerprint_j = fingerprints_2[idx_fingerprint_j, :]
        row_index = idx_fingerprint_j + size_1
        tanimoto_scores = tanimoto_scores_row_dense(fingerprint_j, fingerprints_1)

        _fill_pairs_for_row_between_sets(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            row_index,
            0,
            max_pairs_per_bin,
            selection_bins,
        )

    return selected_pairs_per_bin, selected_scores_per_bin


@jit(nopython=True, parallel=True)
def _compute_tanimoto_similarity_per_bin_between_sets_sparse_binary(
    fingerprints_1,
    fingerprints_2,
    max_pairs_per_bin,
    selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)], dtype=np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    size_1 = len(fingerprints_1)
    size_2 = len(fingerprints_2)
    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size_1 + size_2, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size_1 + size_2, max_pairs_per_bin), dtype=np.float32)

    for idx_fingerprint_i in prange(size_1):
        fingerprint_i = fingerprints_1[idx_fingerprint_i]
        tanimoto_scores = tanimoto_scores_row_sparse_binary(fingerprint_i, fingerprints_2)

        _fill_pairs_for_row_between_sets(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            idx_fingerprint_i,
            size_1,
            max_pairs_per_bin,
            selection_bins,
        )

    for idx_fingerprint_j in prange(size_2):
        fingerprint_j = fingerprints_2[idx_fingerprint_j]
        row_index = idx_fingerprint_j + size_1
        tanimoto_scores = tanimoto_scores_row_sparse_binary(fingerprint_j, fingerprints_1)

        _fill_pairs_for_row_between_sets(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            row_index,
            0,
            max_pairs_per_bin,
            selection_bins,
        )

    return selected_pairs_per_bin, selected_scores_per_bin


@jit(nopython=True, parallel=True)
def _compute_tanimoto_similarity_per_bin_between_sets_sparse_count(
    fingerprints_1_bins,
    fingerprints_1_counts,
    fingerprints_2_bins,
    fingerprints_2_counts,
    max_pairs_per_bin,
    selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)], dtype=np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    size_1 = len(fingerprints_1_bins)
    size_2 = len(fingerprints_2_bins)
    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size_1 + size_2, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size_1 + size_2, max_pairs_per_bin), dtype=np.float32)

    for idx_fingerprint_i in prange(size_1):
        bins_i = fingerprints_1_bins[idx_fingerprint_i]
        counts_i = fingerprints_1_counts[idx_fingerprint_i]
        tanimoto_scores = tanimoto_scores_row_sparse_count(
            bins_i,
            counts_i,
            fingerprints_2_bins,
            fingerprints_2_counts,
        )

        _fill_pairs_for_row_between_sets(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            idx_fingerprint_i,
            size_1,
            max_pairs_per_bin,
            selection_bins,
        )

    for idx_fingerprint_j in prange(size_2):
        bins_j = fingerprints_2_bins[idx_fingerprint_j]
        counts_j = fingerprints_2_counts[idx_fingerprint_j]
        row_index = idx_fingerprint_j + size_1
        tanimoto_scores = tanimoto_scores_row_sparse_count(
            bins_j,
            counts_j,
            fingerprints_1_bins,
            fingerprints_1_counts,
        )

        _fill_pairs_for_row_between_sets(
            selected_pairs_per_bin,
            selected_scores_per_bin,
            tanimoto_scores,
            row_index,
            0,
            max_pairs_per_bin,
            selection_bins,
        )

    return selected_pairs_per_bin, selected_scores_per_bin


def compute_tanimoto_similarity_per_bin_between_sets(
    fingerprints_1,
    fingerprints_2,
    max_pairs_per_bin,
    fingerprint_type: str,
    selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)], dtype=np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cross-set Tanimoto per bin for all supported fingerprint types."""
    if fingerprint_type not in SUPPORTED_FINGERPRINT_TYPES:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")

    if is_dense_fingerprint_type(fingerprint_type):
        return _compute_tanimoto_similarity_per_bin_between_sets_dense(
            fingerprints_1,
            fingerprints_2,
            max_pairs_per_bin=max_pairs_per_bin,
            selection_bins=selection_bins,
        )

    if is_unfolded_binary_fingerprint_type(fingerprint_type):
        return _compute_tanimoto_similarity_per_bin_between_sets_sparse_binary(
            fingerprints_1,
            fingerprints_2,
            max_pairs_per_bin=max_pairs_per_bin,
            selection_bins=selection_bins,
        )

    if is_unfolded_count_fingerprint_type(fingerprint_type):
        fingerprints_1_bins, fingerprints_1_counts = _split_sparse_count_fingerprints(fingerprints_1)
        fingerprints_2_bins, fingerprints_2_counts = _split_sparse_count_fingerprints(fingerprints_2)
        return _compute_tanimoto_similarity_per_bin_between_sets_sparse_count(
            fingerprints_1_bins,
            fingerprints_1_counts,
            fingerprints_2_bins,
            fingerprints_2_counts,
            max_pairs_per_bin=max_pairs_per_bin,
            selection_bins=selection_bins,
        )

    raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
