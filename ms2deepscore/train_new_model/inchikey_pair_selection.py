from collections import Counter
from typing import List, Tuple
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.filtering import add_fingerprint
from matchms.similarity.vector_similarity_functions import jaccard_index
from numba import jit, prange
from tqdm import tqdm
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore

from ms2deepscore.train_new_model import InchikeyPairGenerator


def select_compound_pairs_wrapper(
        spectrums: List[Spectrum],
        settings: SettingsMS2Deepscore,
        ) -> InchikeyPairGenerator:
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

    # Select the nr_of_pairs_per_bin to use
    nr_of_available_pairs_per_bin = get_nr_of_available_pairs_in_bin(available_pairs_per_bin_matrix)
    lowest_max_number_of_pairs = min(nr_of_available_pairs_per_bin) * settings.max_pair_resampling
    print(f"The available nr of pairs per bin are: {nr_of_available_pairs_per_bin}")
    aimed_nr_of_pairs_per_bin = settings.average_pairs_per_bin*len(inchikeys14_unique)
    if lowest_max_number_of_pairs < aimed_nr_of_pairs_per_bin:
        print(f"Warning: The average_pairs_per_bin: {settings.average_pairs_per_bin} cannot be reached, "
              f"since this would require "
              f"{settings.average_pairs_per_bin} * {len(inchikeys14_unique)} = {aimed_nr_of_pairs_per_bin} pairs."
              f"But one of the bins has only {lowest_max_number_of_pairs} available"
              f"Instead the lowest number of available pairs in a bin times the resampling is used, "
              f"which is: {lowest_max_number_of_pairs}")
        aimed_nr_of_pairs_per_bin = lowest_max_number_of_pairs

    pair_frequency_matrixes = balanced_selection_of_pairs_per_bin(available_pairs_per_bin_matrix,
                                                                 settings.max_pair_resampling,
                                                                 aimed_nr_of_pairs_per_bin)

    selected_pairs_per_bin = convert_to_selected_pairs_list(pair_frequency_matrixes, available_pairs_per_bin_matrix,
                                          available_scores_per_bin_matrix, inchikeys14_unique)
    return InchikeyPairGenerator([pair for pairs in selected_pairs_per_bin for pair in pairs])


def convert_to_selected_pairs_list(pair_frequency_matrixes, available_pairs_per_bin_matrix, scores_matrix,
                                   inchikeys14_unique):
    selected_pairs_per_bin = []
    for bin_id, bin_pair_frequency_matrix in enumerate(tqdm(pair_frequency_matrixes)):
        selected_pairs = []
        for inchikey1, pair_frequency_row in enumerate(bin_pair_frequency_matrix):
            for inchikey2_index, pair_frequency in enumerate(pair_frequency_row):
                if pair_frequency > 0:
                    inchikey2 = available_pairs_per_bin_matrix[bin_id][inchikey1][inchikey2_index]
                    score = scores_matrix[bin_id][inchikey1][inchikey2_index]
                    selected_pairs.extend([(inchikeys14_unique[inchikey1], inchikeys14_unique[inchikey2], score)]*pair_frequency)
                    # remove duplicate pairs
                    position_of_first_inchikey_in_matrix = available_pairs_per_bin_matrix[bin_id][inchikey2] == inchikey1
                    bin_pair_frequency_matrix[inchikey2][position_of_first_inchikey_in_matrix] = 0
        selected_pairs_per_bin.append(selected_pairs)
    return selected_pairs_per_bin


def select_balanced_pairs(available_pairs_for_bin_matrix: np.ndarray,
                          inchikey_counts: np.ndarray,
                          required_number_of_pairs: int,
                          max_resampling: int):
    """Select pairs of spectra in a balanced way. """

    nr_of_pairs_selected = 0
    # Keep track of which inchikeys are available in this bin. If all have been sampled it is removed from this list.
    available_inchikey_indexes = list(np.arange(available_pairs_for_bin_matrix.shape[0]))

    # Create a sampling frequency matrix. This matrix keeps track of how frequenctly a pair has been sampled.
    pair_frequency = available_pairs_for_bin_matrix.copy()
    pair_frequency[pair_frequency != -1] = 0
    # All cases where no pair is available is set to max_resampling times 2 (can be any number > max_resampling)
    # This ensures this pair is never selected.
    pair_frequency[pair_frequency == -1] = max_resampling * 2
    with tqdm(total=required_number_of_pairs,
              desc="Balanced sampling of inchikey pairs (will repeat for each bin)") as progress_bar:
        while nr_of_pairs_selected < required_number_of_pairs:
            # get inchikey with lowest count
            inchikey_with_lowest_count = available_inchikey_indexes[
                np.argmin(inchikey_counts[available_inchikey_indexes])]

            # Select the pairs that have been resampled the least frequent.
            lowest_pair_frequency = np.min(pair_frequency[inchikey_with_lowest_count])
            if lowest_pair_frequency >= max_resampling:
                # remove the inchikey, since it does not have any available pairs anymore
                available_inchikey_indexes.remove(inchikey_with_lowest_count)
                if len(available_inchikey_indexes) == 0:
                    raise ValueError("The number of pairs available is less than required_number_of_pairs.")
                continue

            pair_indexes_with_min_count = pair_frequency[inchikey_with_lowest_count] == lowest_pair_frequency

            available_inchikeys_with_min_count = available_pairs_for_bin_matrix[
                inchikey_with_lowest_count][pair_indexes_with_min_count]

            second_inchikey_with_lowest_count = available_inchikeys_with_min_count[np.argmin(
                inchikey_counts[available_inchikeys_with_min_count])]

            # Add the selected pairs to pair_frequency:
            position_of_second_inchikey_in_matrix = available_pairs_for_bin_matrix[
                                                        inchikey_with_lowest_count] == second_inchikey_with_lowest_count
            pair_frequency[inchikey_with_lowest_count][position_of_second_inchikey_in_matrix] += 1

            if second_inchikey_with_lowest_count != inchikey_with_lowest_count:
                # also increase pair_frequency if duplicate
                position_of_first_inchikey_in_matrix = available_pairs_for_bin_matrix[
                                                           second_inchikey_with_lowest_count] == inchikey_with_lowest_count
                pair_frequency[second_inchikey_with_lowest_count][position_of_first_inchikey_in_matrix] += 1
            # increase inchikey counts
            inchikey_counts[inchikey_with_lowest_count] += 1
            inchikey_counts[second_inchikey_with_lowest_count] += 1
            nr_of_pairs_selected += 1
            progress_bar.update(1)
    return pair_frequency, inchikey_counts


def balanced_selection_of_pairs_per_bin(available_pairs_per_bin_matrix: np.ndarray,
                                        max_pair_resampling,
                                        nr_of_pairs_per_bin):
    """From the list_of_pairs_per_bin a balanced selection is made to have a balanced distribution over bins and inchikeys
    """

    inchikey_count = np.zeros(available_pairs_per_bin_matrix.shape[1])
    pair_frequency_matrixes = []
    for pairs_in_bin in available_pairs_per_bin_matrix:
        pair_frequencies, inchikey_count = select_balanced_pairs(pairs_in_bin,
                                                                 inchikey_count,
                                                                 nr_of_pairs_per_bin,
                                                                 max_pair_resampling,
                                                                 )
        pair_frequency_matrixes.append(pair_frequencies)
    pair_frequency_matrixes = np.array(pair_frequency_matrixes)
    pair_frequency_matrixes[pair_frequency_matrixes == 2 * max_pair_resampling] = 0
    return pair_frequency_matrixes


def get_nr_of_available_pairs_in_bin(selected_pairs_per_bin_matrix: np.ndarray) -> List[int]:
    """Calculates the number of unique pairs available per bin, discarding duplicated (inverted) pairs"""
    nr_of_unique_pairs_per_bin = []
    for bin_idx in tqdm(range(selected_pairs_per_bin_matrix.shape[0]),
                        desc="Determining number of available pairs per bin"):
        inchikey_indexes_1, pair_sample_position = np.where(selected_pairs_per_bin_matrix[bin_idx] != -1)
        pairs = []
        for i, inchikey_index_1 in enumerate(inchikey_indexes_1):
            inchikey_index_2 = selected_pairs_per_bin_matrix[bin_idx, inchikey_index_1, pair_sample_position[i]]
            # sort the pairs on inchikey (to later remove duplicates)
            if inchikey_index_1 < inchikey_index_2:
                pairs.append((inchikey_index_1, inchikey_index_2))
            else:
                pairs.append((inchikey_index_2, inchikey_index_1))
        nr_of_unique_pairs_per_bin.append(len(set(pairs)))
    return nr_of_unique_pairs_per_bin


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
