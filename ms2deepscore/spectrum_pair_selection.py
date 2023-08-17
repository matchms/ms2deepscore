from typing import List, Tuple
from collections import Counter
import numba
import numpy as np
from matchms import Spectrum
from matchms.filtering import add_fingerprint
from matchms.similarity.vector_similarity_functions import jaccard_index
from scipy.sparse import coo_array


class SelectedCompoundPairs:
    """Class to store sparse ("cherrypicked") compound pairs and their respective scores.

    This is meant to be used with the results of the `compute_spectrum_pairs()` function.
    The therein selected (cherrypicked) scores are stored similar to a list-of-lists format.

    """
    def __init__(self, coo_array, inchikeys, shuffling: bool = True):
        self._scores = []
        self._cols = []
        self.shuffling = shuffling
        self._idx_to_inchikey = dict(enumerate(inchikeys))
        self._inchikey_to_idx = {key: idx for idx, key in enumerate(inchikeys)}

        for row_idx in self._idx_to_inchikey.keys():
            row_mask = (coo_array.row == row_idx)
            self._cols.append(coo_array.col[row_mask])
            self._scores.append(coo_array.data[row_mask])

        # Initialize counter for each column
        self._row_generator_index = np.zeros(len(self._idx_to_inchikey), dtype=int)
        if self.shuffling:
            self.shuffle()

    def shuffle(self):
        """Shuffle all scores for all inchikeys."""
        for i in range(len(self._scores)):
            self._shuffle_row(i)

    def _shuffle_row(self, row_index):
        """Shuffle the column and scores of row with row_index."""
        permutation = np.random.permutation(len(self._cols[row_index]))
        self._cols[row_index] = self._cols[row_index][permutation]
        self._scores[row_index] = self._scores[row_index][permutation]

    def next_pair_for_inchikey(self, inchikey):
        row_idx = self._inchikey_to_idx[inchikey]

        # Retrieve the next pair
        col_idx = self._cols[row_idx][self._row_generator_index[row_idx]]
        score = self._scores[row_idx][self._row_generator_index[row_idx]]

        # Update the counter, wrapping around if necessary
        self._row_generator_index[row_idx] += 1
        if self._row_generator_index[row_idx] >= len(self._cols[row_idx]):
            self._row_generator_index[row_idx] = 0
            # Went through all scores in this row --> shuffle again
            self._shuffle_row(row_idx)

        return score, self._idx_to_inchikey[col_idx]

    def generator(self):
        """Infinite generator to loop through all inchikeys."""
        while True:
            for inchikey in self._inchikey_to_idx.keys():
                score, inchikey2 = self.next_pair_for_inchikey(inchikey)
                yield inchikey, score, inchikey2

    @property
    def scores(self):
        return self._scores

    @property
    def idx_to_inchikey(self):
        return self._idx_to_inchikey

    @property
    def inchikey_to_idx(self):
        return self._inchikey_to_idx

    def __str__(self):
        return f"SelectedCompoundPairs with {len(self._scores)} columns."


def compute_spectrum_pairs(spectrums,
                           selection_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
                           max_pairs_per_bin: int = 10,
                           include_diagonal: bool = True,
                           fix_global_bias: bool = True,
                           fingerprint_type: str = "daylight",
                           nbits: int = 2048):
    """Function to compute the compound similarities (Tanimoto) and collect a well-balanced set of pairs.

    TODO: describe method and arguments
    """
    # pylint: disable=too-many-arguments
    spectra_selected, inchikeys14_unique = select_inchi_for_unique_inchikeys(spectrums)
    print(f"Selected {len(spectra_selected)} spectra with unique inchikeys (out of {len(spectrums)} spectra)")

    # Compute fingerprints using matchms
    spectra_selected = [add_fingerprint(s, fingerprint_type, nbits)\
                        if s.get("fingerprint") is None else s for s in spectra_selected]

    # Ignore missing / not-computed fingerprints
    fingerprints = [s.get("fingerprint") for s in spectra_selected]
    idx = np.array([i for i, x in enumerate(fingerprints) if x is not None]).astype(int)
    if len(idx) == 0:
        raise ValueError("No fingerprints could be computed")
    if len(idx) < len(fingerprints):
        print(f"Successfully generated fingerprints for {len(idx)} of {len(fingerprints)} spectra")
    fingerprints = [fingerprints[i] for i in idx]
    inchikeys14_unique = [inchikeys14_unique[i] for i in idx]
    spectra_selected = [spectra_selected[i] for i in idx]

    # Compute and return selected scores
    scores_sparse = jaccard_similarity_matrix_cherrypicking(
        np.array(fingerprints),
        selection_bins,
        max_pairs_per_bin,
        include_diagonal,
        fix_global_bias)
    return scores_sparse, inchikeys14_unique #, spectra_selected


def jaccard_similarity_matrix_cherrypicking(
    fingerprints: np.ndarray,
    selection_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
    max_pairs_per_bin: int = 20,
    include_diagonal: bool = True,
    fix_global_bias: bool = True,
    random_seed: int = None,
) -> coo_array:
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
    return coo_array((np.array(data), (np.array(i), np.array(j))),
                      shape=(size, size))


@numba.njit
def compute_jaccard_similarity_per_bin(
        fingerprints: np.ndarray,
        selection_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
        max_pairs_per_bin: int = 20,
        include_diagonal: bool = True) -> List[List[Tuple[int, float]]]:
    """For each inchikey for each bin matches are stored within this bin

    fingerprints
        Fingerprint vectors as 2D numpy array.
    selection_bins
        List of tuples with upper and lower bound for score bins.
        The goal is to pick equal numbers of pairs for each score bin.
        Sidenote: bins do not have to be of equal size, nor do they have to cover the entire
        range of the used scores.
    max_pairs_per_bin
        Specifies the desired maximum number of pairs to be added for each score bin.

    returns:
        A list were the indexes are the bin numbers. This contains Lists were the index is the spectrum_i index.
        This list contains a Tuple, with first the spectrum_j index and second the score.
    """
    # pylint: disable=too-many-locals
    size = fingerprints.shape[0]
    # initialize storing scores
    selected_pairs_per_bin = [[] for _ in range(len(selection_bins))]

    # keep track of total bias across bins
    max_pairs_global = len(selection_bins) * [max_pairs_per_bin]
    for i in range(size):
        scores_row = np.zeros(size)
        for j in range(size):
            if i == j and not include_diagonal:
                continue
            scores_row[j] = jaccard_index(fingerprints[i, :], fingerprints[j, :])

        # Select pairs per bin with a maximum of max_pairs_per_bin
        for bin_number, selection_bin in enumerate(selection_bins):
            selected_pairs_per_bin[bin_number].append([])
            # Indices of scores within the current bin
            idx = np.where((scores_row > selection_bin[0]) & (scores_row <= selection_bin[1]))[0]
            # Randomly select up to max_pairs_per_bin scores within the bin
            np.random.shuffle(idx)
            idx_selected = idx[:max_pairs_global[bin_number]]
            for index in idx_selected:
                selected_pairs_per_bin[bin_number][i].append((index, scores_row[index]))
    return selected_pairs_per_bin


def fix_bias(fingerprints: np.ndarray,
    selection_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
    max_pairs_per_bin: int = 20,
    include_diagonal: bool = True,
    fix_global_bias: bool = True):
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
    if fix_global_bias:
        # todo make the nr of times the max pair is used a variable, with as option "inf" meaning it will store everything it finds for each bin. (for anyone without memory constraints and difficult bins)
        max_pairs_per_bin = max_pairs_per_bin*2
    selected_pairs_per_bin = compute_jaccard_similarity_per_bin(
        fingerprints,
        selection_bins,
        max_pairs_per_bin,
        include_diagonal)
    if not fix_global_bias:
        return selected_pairs_per_bin

    for bin_nr, scores_per_spectrum in enumerate(selected_pairs_per_bin):
        nr_of_pairs_in_bin = []
        for spectrum_i_idx, score_and_idx in enumerate(scores_per_spectrum):
            nr_of_pairs_in_bin.append(len(score_and_idx))
        difference, found_cut_off = find_correct_cut_off(nr_of_pairs_in_bin, max_pairs_per_bin)
        # todo check if this works correctly
        for spectrum_i_idx, score_and_idx in enumerate(scores_per_spectrum):
            if spectrum_i_idx <= difference:
                cut_off = found_cut_off
            else:
                cut_off = found_cut_off - 1
            # Remove excess pairs_per_bin
            selected_pairs_per_bin[bin_nr][spectrum_i_idx] = score_and_idx[:cut_off]
    #todo add a converter function that can convert this to the coo_array function
    return selected_pairs_per_bin


def try_cut_off(nr_of_pairs_in_bin_per_spectrum, cut_off):
    total_nr_of_pairs = 0
    for nr_of_pairs_in_bin in nr_of_pairs_in_bin_per_spectrum:
        if nr_of_pairs_in_bin <= cut_off:
            total_nr_of_pairs += nr_of_pairs_in_bin
        else:
            total_nr_of_pairs += cut_off
    average_nr_of_pairs = total_nr_of_pairs / len(nr_of_pairs_in_bin_per_spectrum)
    return average_nr_of_pairs


def find_correct_cut_off(nr_of_pairs_in_bin_per_spectrum: List[int], expected_average_nr_of_pairs: int):
    found_cut_off = False
    for cut_off in range(expected_average_nr_of_pairs, expected_average_nr_of_pairs*2):
        average_nr_of_pairs = try_cut_off(nr_of_pairs_in_bin_per_spectrum, cut_off)
        if average_nr_of_pairs >= expected_average_nr_of_pairs:
            found_cut_off = cut_off
            break
    assert found_cut_off, "Not enough pairs were found for one of the bins"
    expected_nr_of_pairs = expected_average_nr_of_pairs*len(nr_of_pairs_in_bin_per_spectrum)
    found_nr_of_pairs = average_nr_of_pairs*len(nr_of_pairs_in_bin_per_spectrum)
    # to get the exact number of pairs expected
    difference = found_nr_of_pairs - expected_nr_of_pairs
    return difference, found_cut_off


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
