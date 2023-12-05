from collections import Counter
from typing import List, Optional, Tuple
import numpy as np
from matchms import Spectrum
from matchms.filtering import add_fingerprint
from matchms.similarity.vector_similarity_functions import jaccard_index
from scipy.sparse import coo_array
from tqdm import tqdm
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore


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
        settings: SettingsMS2Deepscore
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

    fingerprints, inchikeys14_unique, spectra_selected = compute_fingerprints_for_training(spectrums,
                                                                                           settings.fingerprint_type,
                                                                                           settings.fingerprint_nbits)

    selected_pairs_per_bin = compute_jaccard_similarity_per_bin(fingerprints,
                                                                settings.tanimoto_bins,
                                                                settings.max_pairs_per_bin,
                                                                settings.include_diagonal)
    selected_pairs_per_bin = fix_bias(selected_pairs_per_bin, settings.average_pairs_per_bin)
    scores_sparse = convert_selected_pairs_per_bin_to_coo_array(selected_pairs_per_bin, fingerprints.shape[0])
    return SelectedCompoundPairs(scores_sparse, inchikeys14_unique), spectra_selected


def convert_selected_pairs_per_bin_to_coo_array(selected_pairs_per_bin: List[List[Tuple[int, float]]], size):
    data = []
    inchikey_indexes_i = []
    inchikey_indexes_j = []
    for scores_per_inchikey in selected_pairs_per_bin:
        assert len(scores_per_inchikey) == size
        for inchikey_idx_i, scores_list in enumerate(scores_per_inchikey):
            for scores in scores_list:
                inchikey_idx_j, score = scores
                data.append(score)
                inchikey_indexes_i.append(inchikey_idx_i)
                inchikey_indexes_j.append(inchikey_idx_j)
    return coo_array((np.array(data), (np.array(inchikey_indexes_i), np.array(inchikey_indexes_j))),
                     shape=(size, size))

# todo refactor so numba.njit can be used again
# @numba.njit
def compute_jaccard_similarity_per_bin(
        fingerprints: np.ndarray,
        selection_bins: np.ndarray = np.array([(x/10, x/10 + 0.1) for x in range(0, 10)]),
        max_pairs_per_bin: Optional[int] = None,
        include_diagonal: bool = True) -> List[List[Tuple[int, float]]]:
    """For each inchikey for each bin matches are stored within this bin.
    The max pairs per bin specifies how many pairs are selected per bin. This helps reduce the memory load.

    fingerprints
        Fingerprint vectors as 2D numpy array.
    selection_bins
        List of tuples with upper and lower bound for score bins.
        The goal is to pick equal numbers of pairs for each score bin.
        Sidenote: bins do not have to be of equal size, nor do they have to cover the entire
        range of the used scores.
    max_pairs_per_bin
        Specifies the desired maximum number of pairs to be added for each score bin.
        Set to None to select everything (more memory intensive)

    returns:
        A list were the indexes are the bin numbers. This contains Lists were the index is the spectrum_i index.
        This list contains a Tuple, with first the spectrum_j index and second the score.
    """
    # pylint: disable=too-many-locals
    size = fingerprints.shape[0]
    # initialize storing scores
    selected_pairs_per_bin = [[] for _ in range(len(selection_bins))]

    # loop over the fingerprints
    for idx_fingerprint_i in range(size):
        fingerprint_i = fingerprints[idx_fingerprint_i, :]

        # Calculate all tanimoto scores for 1 fingerprint
        tanimoto_scores = np.zeros(size)
        for idx_fingerprint_j in range(size):
            if idx_fingerprint_i == idx_fingerprint_j and not include_diagonal:
                # skip matching fingerprint score against itself.
                continue
            fingerprint_j = fingerprints[idx_fingerprint_j, :]
            tanimoto_score = jaccard_index(fingerprint_i, fingerprint_j)
            tanimoto_scores[idx_fingerprint_j] = tanimoto_score

        # Select pairs per bin with a maximum of max_pairs_per_bin
        for bin_number, selection_bin in enumerate(selection_bins):
            selected_pairs_per_bin[bin_number].append([])
            # Indices of scores within the current bin
            idx = np.where((tanimoto_scores > selection_bin[0]) & (tanimoto_scores <= selection_bin[1]))[0]
            # Randomly select up to max_pairs_per_bin scores within the bin
            np.random.shuffle(idx)
            idx_selected = idx[:max_pairs_per_bin]
            for index in idx_selected:
                selected_pairs_per_bin[bin_number][idx_fingerprint_i].append((index, tanimoto_scores[index]))
    return selected_pairs_per_bin


def fix_bias(selected_pairs_per_bin, expected_average_pairs_per_bin):
    """
    Adjusts the selected pairs for each bin to align with the expected average pairs per bin.
    
    This function modifies the number of pairs in each bin to be closer to the 
    expected average pairs per bin by truncating or extending the pairs.
    
    Parameters
    ----------
    selected_pairs_per_bin: list of list
        The list containing bins and for each bin, the list of pairs for each spectrum.
    expected_average_pairs_per_bin: int
        The expected average number of pairs per bin.
    """
    new_selected_pairs_per_bin = []
    for bin_nr, scores_per_compound in enumerate(selected_pairs_per_bin):
        new_selected_pairs_per_bin.append([])
        # Calculate the nr_of_pairs_in_bin_per_compound
        nr_of_pairs_in_bin_per_compound = [len(score_and_idx) for score_and_idx in scores_per_compound]

        cut_offs_to_use = get_nr_of_pairs_needed_to_fix_bias(nr_of_pairs_in_bin_per_compound,
                                                             expected_average_pairs_per_bin)
        if sum(cut_offs_to_use)/len(cut_offs_to_use) != expected_average_pairs_per_bin:
            print(f"For bin {bin_nr} the expected average number of pairs: {expected_average_pairs_per_bin} "
                  f"does not match the actual average number of pairs: {sum(cut_offs_to_use)/len(cut_offs_to_use)}")
        for i, cut_off in enumerate(cut_offs_to_use):
            new_selected_pairs_per_bin[bin_nr].append(scores_per_compound[i][:cut_off])
    return new_selected_pairs_per_bin


def get_nr_of_pairs_needed_to_fix_bias(nr_of_pairs_in_bin_per_compound: List[int],
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
    """Calculates fingerprints for each unique inchikey and removes spectra for which no fingerprint could be created"""
    if len(spectrums) == 0:
        raise ValueError("No spectra were selected to calculate fingerprints")
    spectra_selected, inchikeys14_unique = select_inchi_for_unique_inchikeys(spectrums)
    print(f"Selected {len(spectra_selected)} spectra with unique inchikeys (out of {len(spectrums)} spectra)")
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