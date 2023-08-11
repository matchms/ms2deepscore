import numba
import numpy as np
from matchms.similarity.vector_similarity_functions import jaccard_index
from scipy.sparse import coo_array, lil_array


def jaccard_similarity_matrix_cherrypicking(
    fingerprints: np.ndarray,
    selections_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
    max_pairs_per_bin: int = 20,
    include_diagonal: bool = True,
    fix_global_bias: bool = True,
    random_seed: int = None,
) -> np.ndarray:
    """Returns matrix of jaccard indices between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    fingerprints
        Fingerprint vectors as 2D numpy array.

    Returns
    -------
    scores
        Sparse array (List of lists) with cherrypicked scores.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    size = fingerprints.shape[0]
    if random_seed is not None:
        np.random.seed(random_seed)
    data, i, j = compute_jaccard_similarity_matrix_cherrypicking(
        fingerprints,
        selections_bins,
        max_pairs_per_bin,
        include_diagonal,
        fix_global_bias,
    )
    scores_sparse = coo_array((np.array(data), (np.array(i), np.array(j))), shape=(size, size))
    return lil_array(scores_sparse)


@numba.njit
def compute_jaccard_similarity_matrix_cherrypicking(
    fingerprints: np.ndarray,
    selections_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
    max_pairs_per_bin: int = 20,
    include_diagonal: bool = True,
    fix_global_bias: bool = True,
) -> np.ndarray:
    """Returns matrix of jaccard indices between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    fingerprints
        Fingerprint vectors as 2D numpy array.

    Returns
    -------
    scores
        Sparse array (List of lists) with cherrypicked scores.
    """
    size = fingerprints.shape[0]
    scores_data = []
    scores_i = []
    scores_j = []
    # keep track of total bias across bins
    max_pairs_global = len(selections_bins) * [max_pairs_per_bin]
    for i in range(size):
        scores_row = np.zeros(size)
        for j in range(size):
            if i == j and not include_diagonal:
                continue
            scores_row[j] = jaccard_index(fingerprints[i, :], fingerprints[j, :])
        
        # Cherrypicking
        for bin_number, selection_bin in enumerate(selections_bins):
            # Indices of scores within the current bin
            idx = np.where((scores_row > selection_bin[0]) & (scores_row <= selection_bin[1]))[0]

            # Randomly select up to max_pairs_per_bin scores within the bin
            #if len(idx) > 0:
            np.random.shuffle(idx)
            if fix_global_bias:
                idx_selected = idx[:max_pairs_global[bin_number]]
                max_pairs_global[bin_number] += max_pairs_per_bin
                max_pairs_global[bin_number] -= len(idx_selected)  # only remove actually found pairs
            else:
                idx_selected = idx[:max_pairs_per_bin]
            scores_data.extend(scores_row[idx_selected])
            scores_i.extend(len(idx_selected) * [i])
            scores_j.extend(list(idx_selected))
    print(max_pairs_global)
        
    return scores_data, scores_i, scores_j
