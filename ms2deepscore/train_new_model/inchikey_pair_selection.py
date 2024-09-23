import logging
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
import json


class SelectedInchikeyPairs:
    def __init__(self, selected_inchikey_pairs: List[Tuple[str, str, float]]):
        """
        Parameters
        ----------
        selected_inchikey_pairs:
            A list with tuples encoding inchikey pairs like: (inchikey1, inchikey2, tanimoto_score)
        """
        self.selected_inchikey_pairs = selected_inchikey_pairs

    def generator(self, shuffle: bool, random_nr_generator):
        """Infinite generator to loop through all inchikeys.
        After looping through all inchikeys the order is shuffled.
        """
        while True:
            if shuffle:
                random_nr_generator.shuffle(self.selected_inchikey_pairs)

            for inchikey1, inchikey2, tanimoto_score in self.selected_inchikey_pairs:
                yield inchikey1, inchikey2, tanimoto_score

    def __len__(self):
        return len(self.selected_inchikey_pairs)

    def __str__(self):
        return f"SelectedInchikeyPairs with {len(self.selected_inchikey_pairs)} pairs available"

    def get_scores(self):
        return [score for _, _, score in self.selected_inchikey_pairs]

    def get_inchikey_counts(self) -> Counter:
        """returns the frequency each inchikey occurs"""
        inchikeys = Counter()
        for inchikey_1, inchikey_2, _ in self.selected_inchikey_pairs:
            inchikeys[inchikey_1] += 1
            inchikeys[inchikey_2] += 1
        return inchikeys

    def get_scores_per_inchikey(self):
        inchikey_scores = {}
        for inchikey_1, inchikey_2, score in self.selected_inchikey_pairs:
            if inchikey_1 in inchikey_scores:
                inchikey_scores[inchikey_1].append(score)
            else:
                inchikey_scores[inchikey_1] = []
            if inchikey_2 in inchikey_scores:
                inchikey_scores[inchikey_2].append(score)
            else:
                inchikey_scores[inchikey_2] = []
        return inchikey_scores

    def save_as_json(self, file_name):
        data_for_json = [(item[0], item[1], float(item[2])) for item in self.selected_inchikey_pairs]

        with open(file_name, "w") as f:
            json.dump(data_for_json, f)


def select_compound_pairs_wrapper(
        spectrums: List[Spectrum],
        settings: SettingsMS2Deepscore,
        ) -> SelectedInchikeyPairs:
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
    fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
        spectrums,
        settings.fingerprint_type,
        settings.fingerprint_nbits)

    available_pairs_per_bin_matrix, available_scores_per_bin_matrix = compute_jaccard_similarity_per_bin(
        fingerprints,
        settings.max_pairs_per_bin,
        settings.same_prob_bins,
        settings.include_diagonal)

    available_pairs_per_bin = convert_selected_pairs_matrix(available_pairs_per_bin_matrix,
                                                            available_scores_per_bin_matrix, inchikeys14_unique)

    # Select the nr_of_pairs_per_bin to use
    nr_of_pairs_per_bin = settings.average_pairs_per_bin*len(inchikeys14_unique)
    lowest_max_number_of_pairs = min(len(pairs) for pairs in available_pairs_per_bin) * settings.max_pair_resampling
    if lowest_max_number_of_pairs < nr_of_pairs_per_bin:
        nr_of_pairs_per_bin = lowest_max_number_of_pairs
        print("Warning: The set average_pairs_per_bin cannot be reached. "
              "Instead the lowest number of available pairs in a bin times the resampling is used")

    selected_pairs_per_bin = balanced_selection_of_pairs_per_bin(available_pairs_per_bin, inchikeys14_unique,
                                                                 settings, nr_of_pairs_per_bin)
    return SelectedInchikeyPairs([pair for pairs in selected_pairs_per_bin for pair in pairs])


def convert_selected_pairs_matrix(selected_pairs_per_bin_matrix, scores_per_bin, inchikeys) -> List[List[Tuple[str, str, float]]]:
    """Converts the matrix with pairs and the matrix with the corresponding scores into lists of pairs.
    A pair is encoded as a tuple(inchikey1, inchikey2, score).
    Any repeating pairs are removed, including inversed pairs."""
    selected_pairs_per_bin = []
    for bin_idx in range(selected_pairs_per_bin_matrix.shape[0]):
        inchikey_indexes_1, pair_sample_position = np.where(selected_pairs_per_bin_matrix[bin_idx] != -1)
        pairs = []
        for i, inchikey_index_1 in enumerate(inchikey_indexes_1):
            inchikey_1 = inchikeys[inchikey_index_1]
            inchikey_2 = inchikeys[selected_pairs_per_bin_matrix[bin_idx, inchikey_index_1, pair_sample_position[i]]]
            score = scores_per_bin[bin_idx, inchikey_index_1, pair_sample_position[i]]
            # sort the pairs on inchikey (to later remove duplicates)
            if inchikey_1 < inchikey_2:
                pairs.append((inchikey_1, inchikey_2, score))
            else:
                pairs.append((inchikey_2, inchikey_1, score))
        # Remove duplicates
        pairs = list(set(pairs))
        selected_pairs_per_bin.append(pairs)
    return selected_pairs_per_bin


def balanced_selection_of_pairs_per_bin(list_of_pairs_per_bin,
                                        unique_inchikeys,
                                        settings,
                                        nr_of_pairs_per_bin):
    """From the list_of_pairs_per_bin a balanced selection is made to have a balanced distribution over bins and inchikeys
    """

    inchikey_count = {inchikey: 0 for inchikey in unique_inchikeys}
    selected_pairs_per_bin = []
    for pairs_in_bin in tqdm(list_of_pairs_per_bin):
        selected_pairs, inchikey_count = select_balanced_pairs(pairs_in_bin,
                                                               inchikey_count,
                                                               nr_of_pairs_per_bin,
                                                               settings.max_pair_resampling)
        selected_pairs_per_bin.append(selected_pairs)
    return selected_pairs_per_bin


def select_balanced_pairs(list_of_available_pairs: List[Tuple[str, str, float]],
                          inchikey_counts: dict,
                          required_number_of_pairs: int,
                          max_resampling: int):
    """Select pairs of spectra in a balanced way. """

    selected_pairs = []
    # Select only the inchikeys that have a pair available for this bin.
    available_inchikey_indexes = get_available_inchikey_indexes(list_of_available_pairs)
    # Store the frequency each pair has been sampled for keeping track of resampling
    pair_frequency = {pair: 0 for pair in list_of_available_pairs}

    while len(selected_pairs) < required_number_of_pairs:
        # get inchikey with lowest count
        inchikey_with_lowest_count = get_available_inchikeys_with_lowest_count(available_inchikey_indexes,
                                                                               inchikey_counts)
        # actually select pairs (instead of single inchikeys)
        available_pairs_for_least_frequent_inchikey = select_available_pairs(list_of_available_pairs,
                                                                             inchikey_with_lowest_count)
        # Select the pairs that have been resampled the least frequent.
        available_pairs_with_least_frequency = select_least_frequent_pairs(available_pairs_for_least_frequent_inchikey,
                                                                           pair_frequency,
                                                                           max_resampling)

        if available_pairs_with_least_frequency is None:
            # remove the inchikey, since it does not have any available pairs anymore
            available_inchikey_indexes.remove(inchikey_with_lowest_count)
            if len(available_inchikey_indexes) == 0:
                raise ValueError("The number of pairs available is less than required_number_of_pairs.")
            continue
        idx_1, idx_2, score = select_second_least_frequent_inchikey(inchikey_counts,
                                                                    available_pairs_with_least_frequency,
                                                                    inchikey_with_lowest_count)

        # Add the selected pair
        selected_pairs.append((idx_1, idx_2, score))

        # Increase count of pair and inchikeys
        pair_frequency[(idx_1, idx_2, score)] += 1
        inchikey_counts[idx_1] += 1
        inchikey_counts[idx_2] += 1
    return selected_pairs, inchikey_counts


def select_second_least_frequent_inchikey(inchikey_counts, available_pairs, least_frequent_inchikey):
    second_inchikey_count=[]
    for inchikey_1, inchikey_2, score in available_pairs:
        if inchikey_1 == least_frequent_inchikey:
            other_inchikey = inchikey_2
        else:
            other_inchikey = inchikey_1

        second_inchikey_count.append(inchikey_counts[other_inchikey])

    index_of_least_frequent_second_index = second_inchikey_count.index(min(second_inchikey_count))
    return available_pairs[index_of_least_frequent_second_index]


def get_available_inchikeys_with_lowest_count(available_inchikey_indexes: set, inchikey_counts: dict):
    # Select only the counts of the available_inchikey_indexes
    available_inchikey_counts = {inchikey: count for inchikey, count in inchikey_counts.items() if
                                 inchikey in available_inchikey_indexes}
    minimum_inchikey_frequency = min(list(available_inchikey_counts.values()))
    least_frequent_inchikeys = [key for key, count in available_inchikey_counts.items() if
                                count == minimum_inchikey_frequency]
    return least_frequent_inchikeys[0]


def select_available_pairs(available_inchikey_pairs: List[Tuple[str, str, float]], least_occuring_inchikey: str):
    """Searches for available pairs"""
    pairs_matching_inchikey = []
    for pair in available_inchikey_pairs:
        idx_1, idx_2, score = pair
        if least_occuring_inchikey == idx_1 or least_occuring_inchikey == idx_2:
            pairs_matching_inchikey.append((idx_1, idx_2, score))

    if len(pairs_matching_inchikey) == 0:
        raise ValueError("select_available_pair expects a inchikey_idx_of_interst that is available in list_of_pairs")
    return pairs_matching_inchikey


def select_least_frequent_pairs(selected_pairs: List[Tuple[str, str, float]], pair_counts: dict, max_resampling: int):
    """Selects the pairs with the lowest frequency"""
    frequency_of_pairs = [pair_counts[selected_pair] for selected_pair in selected_pairs]
    lowest_frequency_of_pairs = min(frequency_of_pairs)
    if lowest_frequency_of_pairs >= max_resampling:
        return None
    pairs_with_lowest_frequency = [selected_pairs[i] for i, frequency in enumerate(frequency_of_pairs) if
                                   frequency == lowest_frequency_of_pairs]
    return pairs_with_lowest_frequency


def get_available_inchikey_indexes(list_of_pairs) -> set:
    available_inchikeys = []
    for inchikey_1, inchikey_2, _ in list_of_pairs:
        available_inchikeys.append(inchikey_1)
        available_inchikeys.append(inchikey_2)
    return set(available_inchikeys)


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


def convert_pair_list_to_coo_array(selected_pairs: List[Tuple[int, int, float]], size):
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
        selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)]),
        include_diagonal=True) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly selects compound pairs per tanimoto bin, up to max_pairs_per_bin

    returns:
    2 3d numpy arrays are returned, the first encodes the pairs per bin and the second the corresponding scores.
    A 3D numpy array with shape [nr_of_bins, nr_of_fingerprints, max_pairs_per_bin].
    An example structure for bin 1, with 3 fingerprints and max_pairs_per_bin =4 would be:
    [[1,2,-1,-1],
    [0,3,-1,-1],
    [0,2,-1,-1],]
    The pairs are encoded by the index and the value.
    So the first row encodes pairs between fingerpint 0 and 1, fingerprint 0 and 2.
    The -1 encode that no more pairs were found for this fingerprint in this bin.
    """

    size = fingerprints.shape[0]
    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size, max_pairs_per_bin), dtype=np.float32)

    # pylint: disable=not-an-iterable
    for idx_fingerprint_i in prange(size):
        tanimoto_scores = tanimoto_scores_row(fingerprints, idx_fingerprint_i)

        for bin_number in range(num_bins):
            selection_bin = selection_bins[bin_number]
            indices = np.nonzero((tanimoto_scores > selection_bin[0]) & (tanimoto_scores <= selection_bin[1]))[0]

            if not include_diagonal and idx_fingerprint_i in indices:
                indices = indices[indices != idx_fingerprint_i]

            np.random.shuffle(indices)
            indices = indices[:max_pairs_per_bin]
            num_indices = len(indices)
 
            selected_pairs_per_bin[bin_number, idx_fingerprint_i, :num_indices] = indices
            selected_scores_per_bin[bin_number, idx_fingerprint_i, :num_indices] = tanimoto_scores[indices]

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

def compute_fingerprint_dataframe(
        spectrums: List[Spectrum],
        fingerprint_type,
        fingerprint_nbits,
        ) -> pd.DataFrame:
    """Returns a dataframe with a fingerprints dataframe

    spectrums:
        A list of spectra
    settings:
        The settings that should be used for selecting the compound pairs wrapper. The settings should be specified as a
        SettingsMS2Deepscore object.
    """
    fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
        spectrums,
        fingerprint_type,
        fingerprint_nbits)

    fingerprints_df = pd.DataFrame(fingerprints, index=inchikeys14_unique)
    return fingerprints_df


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
    return fingerprints, inchikeys14_unique


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
