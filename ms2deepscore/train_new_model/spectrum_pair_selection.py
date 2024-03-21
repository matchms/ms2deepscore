from collections import Counter
from typing import List, Tuple
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.filtering import add_fingerprint
from matchms.similarity.vector_similarity_functions import jaccard_index
from numba import jit, prange
from scipy.sparse import coo_array
from tqdm import tqdm
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore


class SelectedCompoundPairs:
    """Class to store sparse ("cherrypicked") compound pairs and their respective scores.

    This is meant to be used with the results of the `compute_spectrum_pairs()` function.
    The therein selected (cherrypicked) scores are stored similar to a list-of-lists format.

    """
    def __init__(self, sparse_score_array, inchikeys, shuffling: bool = True):
        """
        Parameters
        ----------
        sparse_score_array
            Scipy COO-style sparse array which stores the similarity scores.
            Meant to be used with the results of the compute_spectrum_pairs() function.
        inchikeys
            List or Array of the inchikeys in the order of the sparse_score_array.
            Meant to be used with the results of the compute_spectrum_pairs() function.
        shuffling
            Default is True in which case the selected pairs for each inchikey will be
            shuffled.
        """
        self._scores = []
        self._cols = []
        self.shuffling = shuffling
        self._idx_to_inchikey = dict(enumerate(inchikeys))
        self._inchikey_to_idx = {key: idx for idx, key in enumerate(inchikeys)}

        for row_idx in self._idx_to_inchikey.keys():
            row_mask = (sparse_score_array.row == row_idx)
            self._cols.append(sparse_score_array.col[row_mask])
            self._scores.append(sparse_score_array.data[row_mask])

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
            if self.shuffling:
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


def select_compound_pairs_wrapper(
        spectrums: List[Spectrum],
        settings: SettingsMS2Deepscore,
        shuffling: bool = True,
        ) -> Tuple[SelectedCompoundPairs, List[Spectrum]]:
    """Returns a SelectedCompoundPairs object containing equally balanced pairs over the different bins

    spectrums:
        A list of spectra
    settings:
        The settings that should be used for selecting the compound pairs wrapper. The settings should be specified as a
        SettingsMS2Deepscore object.

    Returns
    -------
    scores
        Sparse array (List of lists) with cherrypicked scores.
    """
    if settings.random_seed is not None:
        np.random.seed(settings.random_seed)

    fingerprints, inchikeys14_unique, spectra_selected = compute_fingerprints_for_training(
        spectrums,
        settings.fingerprint_type,
        settings.fingerprint_nbits)

    selected_pairs_per_bin, selected_scores_per_bin = compute_jaccard_similarity_per_bin(
        fingerprints,
        settings.max_pairs_per_bin,
        settings.same_prob_bins,
        settings.include_diagonal)
    selected_pairs_per_bin = balanced_selection(
        selected_pairs_per_bin,
        selected_scores_per_bin,
        fingerprints.shape[0] * settings.average_pairs_per_bin)
    scores_sparse = convert_pair_list_to_coo_array(selected_pairs_per_bin, fingerprints.shape[0])
    return SelectedCompoundPairs(scores_sparse, inchikeys14_unique, shuffling=shuffling), spectra_selected


def compute_fingerprint_dataframe(
        spectrums: List[Spectrum],
        fingerprint_type,
        fingerprint_nbits,
        ) -> pd.DataFrame:
    """Returns a SelectedCompoundPairs object containing equally balanced pairs over the different bins

    spectrums:
        A list of spectra
    settings:
        The settings that should be used for selecting the compound pairs wrapper. The settings should be specified as a
        SettingsMS2Deepscore object.
    """
    fingerprints, inchikeys14_unique, _ = compute_fingerprints_for_training(
        spectrums,
        fingerprint_type,
        fingerprint_nbits)

    fingerprints_df = pd.DataFrame(fingerprints, index=inchikeys14_unique)
    return fingerprints_df


def convert_pair_array_to_coo_data(
        selected_pairs_per_bin, selected_scores_per_bin):
    data = []
    inchikey_indexes_i = []
    inchikey_indexes_j = []
    for row_id in range(selected_pairs_per_bin.shape[1]):
        idx = np.where(selected_pairs_per_bin[:, row_id, :] != -1)
        data.extend(selected_scores_per_bin[idx[0], row_id, idx[1]])
        inchikey_indexes_i.extend(row_id * np.ones(len(idx[0])))
        inchikey_indexes_j.extend(selected_pairs_per_bin[idx[0], row_id, idx[1]])
    return np.array(data), np.array(inchikey_indexes_i), np.array(inchikey_indexes_j)


def convert_pair_array_to_coo_array(
        selected_pairs_per_bin, selected_scores_per_bin, size):
    data, inchikey_indexes_i, inchikey_indexes_j = convert_pair_array_to_coo_data(
        selected_pairs_per_bin, selected_scores_per_bin)
    return coo_array((data, (inchikey_indexes_i, np.array(inchikey_indexes_j))),
                     shape=(size, size))


def convert_pair_list_to_coo_array(selected_pairs: List[List[Tuple[int, float]]], size):
    data = []
    inchikey_indexes_i = []
    inchikey_indexes_j = []
    for inchikey_idx_i, inchikey_idx_j, score in selected_pairs:
        data.append(score)
        inchikey_indexes_i.append(inchikey_idx_i)
        inchikey_indexes_j.append(inchikey_idx_j)
    return coo_array((np.array(data), (np.array(inchikey_indexes_i), np.array(inchikey_indexes_j))),
                     shape=(size, size))


@jit(nopython=True, parallel=True)
def compute_jaccard_similarity_per_bin(
        fingerprints,
        max_pairs_per_bin,
        selection_bins = np.array([(x / 10, x / 10 + 0.1) for x in range(10)]),
        include_diagonal = True):
    """Randomly selects compound pairs per tanimoto bin, up to max_pairs_per_bin"""
    
    size = fingerprints.shape[0]
    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size, max_pairs_per_bin), dtype=np.float32)
    # pylint: disable=not-an-iterable
    for idx_fingerprint_i in prange(size):
        tanimoto_scores = tanimoto_scores_row(fingerprints, idx_fingerprint_i)

        for bin_number in range(num_bins):
            selection_bin = selection_bins[bin_number]
            if bin_number == 0:
                indices = np.nonzero((tanimoto_scores >= selection_bin[0]) & (tanimoto_scores <= selection_bin[1]))[0]
            else:
                indices = np.nonzero((tanimoto_scores > selection_bin[0]) & (tanimoto_scores <= selection_bin[1]))[0]
            if not include_diagonal and idx_fingerprint_i in indices:
                indices = np.delete(indices, idx_fingerprint_i)
            np.random.shuffle(indices)
            indices = indices[:max_pairs_per_bin]

            selected_pairs_per_bin[bin_number, idx_fingerprint_i, :len(indices)] = indices
            selected_scores_per_bin[bin_number, idx_fingerprint_i, :len(indices)] = tanimoto_scores[indices]

    return selected_pairs_per_bin, selected_scores_per_bin


@jit(nopython=True)
def tanimoto_scores_row(fingerprints, idx):
    size = fingerprints.shape[0]
    tanimoto_scores = np.zeros(size)

    fingerprint_i = fingerprints[idx, :]
    for idx_fingerprint_j in range(size):
        fingerprint_j = fingerprints[idx_fingerprint_j, :]
        tanimoto_score = jaccard_index(fingerprint_i, fingerprint_j)
        tanimoto_scores[idx_fingerprint_j] = tanimoto_score
    return tanimoto_scores


def balanced_selection(selected_pairs_per_bin,
                       selected_scores_per_bin,
                       desired_pairs_per_bin,
                       max_oversampling_rate: float = 1):
    """
    Adjusts the selected pairs for each bin to align with the expected average pairs per bin.
    
    This function modifies the number of pairs in each bin to be closer to the 
    expected average pairs per bin by truncating or extending the pairs.
    
    Parameters
    ----------
    selected_pairs_per_bin: list of list
        The list containing bins and for each bin, the list of pairs for each spectrum.
    desired_pairs_per_bin: int
        The desired number of pairs per bin. Will be used if sufficient scores in each bin are found.
    max_oversampling_rate: float
        Maximum factor for oversampling. This will allow for sampling the same pairs multiple times in the same bin
        to reach the desired_pairs_per_bin.
    """
    if max_oversampling_rate != 1:
        raise NotImplementedError("oversampling is not yet supported")
    # The selected pairs per bin is set to -1 if something is not a pair
    available_pairs = (selected_pairs_per_bin[:, :, :] != -1).sum(axis=2).sum(axis=1)
    minimum_bin_occupation = available_pairs.min()
    print(f"Found minimum bin occupation of {minimum_bin_occupation} pairs.")
    print(f"Bin occupations are: {available_pairs}.")
    
    pairs_per_bin = min(minimum_bin_occupation * max_oversampling_rate, desired_pairs_per_bin)
    if desired_pairs_per_bin > minimum_bin_occupation * max_oversampling_rate:
        print(f"The desired number of {desired_pairs_per_bin} pairs per bin cannot be reached with the current setting.")
        print(f"The number of pairs per bin will be set to {minimum_bin_occupation * max_oversampling_rate}.")
    
    new_selected_pairs_per_bin = []
    
    for bin_id in range(selected_pairs_per_bin.shape[0]):
        goal = pairs_per_bin
        for _ in range(int(np.ceil(max_oversampling_rate))):
            # We loop over the columns. The 100 columns contain the matches from left to right. So if only 10 matches
            # are found in this bin. The first 10 columns are filled and the rest is -1. By starting from the first
            # column, we are first using all compounds that have the lowest number in this bin and only use some
            # compounds extra if necessary.
            for col in range(selected_pairs_per_bin.shape[2]):
                idx = np.where(selected_pairs_per_bin[bin_id, :, col] != -1)[0]
                if len(idx) > goal:
                    idx = np.random.choice(idx, goal)
                if len(idx) == 0 and goal > 0:
                    print(f"Apply oversampling for bin {bin_id}.")
                    break
                goal -= len(idx)
                pairs = [(idx[i], selected_pairs_per_bin[bin_id, idx[i], col],
                          selected_scores_per_bin[bin_id, idx[i], col]) for i in range(len(idx))]
                new_selected_pairs_per_bin.extend(pairs)
                if goal <= 0:
                    break
    return new_selected_pairs_per_bin


def get_nr_of_pairs_needed_to_balanced_selection(nr_of_pairs_in_bin_per_compound: List[int],
                                       expected_average_pairs_per_bin: int
                                       ):
    """Calculates how many pairs should be selected to get the exact number o """
    used_cut_offs = nr_of_pairs_in_bin_per_compound[:]
    while expected_average_pairs_per_bin < sum(used_cut_offs)/len(used_cut_offs):
        used_cut_offs[used_cut_offs.index(max(used_cut_offs))] -= 1
    return used_cut_offs


def compute_fingerprints_for_training(spectrums,
                                      fingerprint_type: str = "daylight",
                                      nbits: int = 2048):
    """Calculates fingerprints for each unique inchikey.
    
    Function also removes spectra for which no fingerprint could be created.
    
    Parameters
    ----------
    fingerprint_type:
        The fingerprint type that should be used for tanimoto score calculations.
    fingerprint_nbits:
        The number of bits to use for the fingerprint.
    """
    if len(spectrums) == 0:
        raise ValueError("No spectra were selected to calculate fingerprints")

    spectra_selected, inchikeys14_unique = select_inchi_for_unique_inchikeys(spectrums)
    print(f"Selected {len(spectra_selected)} spectra with unique inchikeys for calculating tanimoto scores "
          f"(out of {len(spectrums)} spectra)")

    # Compute fingerprints using matchms
    spectra_selected = [add_fingerprint(s, fingerprint_type, nbits)\
                        if s.get("fingerprint") is None else s for s in spectra_selected]

    # Ignore missing / not-computed fingerprints
    fingerprints = [s.get("fingerprint") for s in tqdm(spectra_selected,
                                                       desc="Calculating fingerprints")]
    idx = np.array([i for i, x in enumerate(fingerprints) if x is not None]).astype(int)
    if len(idx) == 0:
        raise ValueError("No fingerprints could be computed")
    if len(idx) < len(fingerprints):
        print(f"Successfully generated fingerprints for {len(idx)} of {len(fingerprints)} spectra")

    fingerprints = np.array([fingerprints[i] for i in idx])
    inchikeys14_unique = [inchikeys14_unique[i] for i in idx]
    spectra_selected = [spectra_selected[i] for i in idx]
    return fingerprints, inchikeys14_unique, spectra_selected


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
