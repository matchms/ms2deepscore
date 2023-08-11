from typing import List, Tuple
from collections import Counter
import numba
import numpy as np
from matchms.similarity.vector_similarity_functions import jaccard_index
from scipy.sparse import coo_array, lil_array


def compute_spectrum_pairs(spectrums):
    pass


def select_inchi_for_unique_inchikeys(
    list_of_spectra: List['Spectrum']
) -> Tuple[List['Spectrum'], List[str]]:
    """Select spectra with most frequent inchi for unique inchikeys.

    Method needed to calculate Tanimoto scores.
    """
    # Extract inchi's and inchikeys from spectra metadata
    inchikeys_list = [s.get("inchikey") for s in list_of_spectra]
    inchi_list = [s.get("inchi") for s in list_of_spectra]

    inchi_array = np.array(inchi_list)
    inchikeys14_array = np.array([x[:14] for x in inchikeys_list])

    # Find unique inchikeys
    inchikeys14_unique = sorted(set(inchikeys14_array))

    spectra_selected = []
    for inchikey14 in inchikeys14_unique:
        # Indices of matching inchikeys
        idx = np.where(inchikeys14_array == inchikey14)[0]

        # Find the most frequent inchi for the inchikey
        most_common_inchi = Counter(inchi_array[idx]).most_common(1)[0][0]

        # ID of the spectrum with the most frequent inchi
        ID = idx[np.where(inchi_array[idx] == most_common_inchi)[0][0]]

        spectra_selected.append(list_of_spectra[ID].clone())

    return spectra_selected, inchikeys14_unique


def jaccard_similarity_matrix_cherrypicking(
    fingerprints: np.ndarray,
    selection_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
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
    selection_bins
        List of tuples with upper and lower bound for score bins.
        The goal is to pick equal numbers of pairs for each score bin.
        Sidenote: bins do not have to be of equal size, nor do they have to cover the entire
        range of the used scores.
    max_pairs_per_bin
        Specifies the desired maximum number of pairs to be added for each score bin.
    include_diagonal
        Set to False if pairs with two equal compounds/fingerprints should be excluded.
    fix_global_bias
        Default is True in which case the function aims to get the same amount of pairs for
        each bin globally. This means it add more than max_pairs_par_bin for some bins and/or
        some compounds to compensate for lack of such scores in other compounds.
    random_seed
        Set to integer if the randomness of the pair selection should be reproducible.

    Returns
    -------
    scores
        Sparse array (List of lists) with cherrypicked scores.
    """
    # pylint: disable=too-many-arguments
    size = fingerprints.shape[0]
    if random_seed is not None:
        np.random.seed(random_seed)
    data, i, j = compute_jaccard_similarity_matrix_cherrypicking(
        fingerprints,
        selection_bins,
        max_pairs_per_bin,
        include_diagonal,
        fix_global_bias,
    )
    scores_sparse = coo_array((np.array(data), (np.array(i), np.array(j))), shape=(size, size))
    return lil_array(scores_sparse)


@numba.njit
def compute_jaccard_similarity_matrix_cherrypicking(
    fingerprints: np.ndarray,
    selection_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
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
    selection_bins
        List of tuples with upper and lower bound for score bins.
        The goal is to pick equal numbers of pairs for each score bin.
        Sidenote: bins do not have to be of equal size, nor do they have to cover the entire
        range of the used scores.
    max_pairs_per_bin
        Specifies the desired maximum number of pairs to be added for each score bin.
    include_diagonal
        Set to False if pairs with two equal compounds/fingerprints should be excluded.
    fix_global_bias
        Default is True in which case the function aims to get the same amount of pairs for
        each bin globally. This means it add more than max_pairs_par_bin for some bins and/or
        some compounds to compensate for lack of such scores in other compounds.

    Returns
    -------
    scores
        Sparse array (List of lists) with cherrypicked scores.
    """
    # pylint: disable=too-many-locals
    size = fingerprints.shape[0]
    scores_data = []
    scores_i = []
    scores_j = []
    # keep track of total bias across bins
    max_pairs_global = len(selection_bins) * [max_pairs_per_bin]
    for i in range(size):
        scores_row = np.zeros(size)
        for j in range(size):
            if i == j and not include_diagonal:
                continue
            scores_row[j] = jaccard_index(fingerprints[i, :], fingerprints[j, :])
        
        # Cherrypicking
        for bin_number, selection_bin in enumerate(selection_bins):
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
