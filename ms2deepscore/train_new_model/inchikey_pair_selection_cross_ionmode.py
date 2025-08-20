from typing import List, Tuple
import numpy as np
from matchms import Spectrum
from numba import jit, prange
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.inchikey_pair_selection import compute_fingerprints_for_training, \
    balanced_selection_of_pairs_per_bin, convert_to_selected_pairs_list, tanimoto_scores_row


def select_compound_pairs_wrapper_across_ionmode(
        spectra_1: List[Spectrum],
        spectra_2: List[Spectrum],
        settings: SettingsMS2Deepscore,
) -> List[Tuple[str, str, float]]:
    """Returns a InchikeyPairGenerator object containing equally balanced pairs over the different bins

    spectra:
        A list of spectra
    settings:
        The settings that should be used for selecting the compound pairs wrapper. The settings should be specified as a
        SettingsMS2Deepscore object.

    Returns
    -------
    InchikeyPairGenerator
        InchikeyPairGenerator containing balanced pairs. The pairs are stored as [(inchikey1, inchikey2, score)]
    """
    if settings.random_seed is not None:
        np.random.seed(settings.random_seed)

    fingerprints_1, inchikeys14_unique_1 = compute_fingerprints_for_training(
        spectra_1,
        settings.fingerprint_type,
        settings.fingerprint_nbits
        )
    fingerprints_2, inchikeys14_unique_2 = compute_fingerprints_for_training(
        spectra_2,
        settings.fingerprint_type,
        settings.fingerprint_nbits
        )

    if len(inchikeys14_unique_1) < settings.batch_size or len(inchikeys14_unique_2) < settings.batch_size:
        raise ValueError("The number of unique inchikeys must be larger than the batch size.")

    available_pairs_per_bin_matrix, available_scores_per_bin_matrix = compute_jaccard_similarity_per_bin_across_ionmodes(
        fingerprints_1, fingerprints_2, settings.max_pairs_per_bin, settings.same_prob_bins)

    pair_frequency_matrixes = balanced_selection_of_pairs_per_bin(
        available_pairs_per_bin_matrix, settings)

    selected_pairs_per_bin = convert_to_selected_pairs_list(
        pair_frequency_matrixes, available_pairs_per_bin_matrix,
        available_scores_per_bin_matrix, inchikeys14_unique_1 + inchikeys14_unique_2)
    return [pair for pairs in selected_pairs_per_bin for pair in pairs]


@jit(nopython=True, parallel=True)
def compute_jaccard_similarity_per_bin_across_ionmodes(
        fingerprints_1,
        fingerprints_2,
        max_pairs_per_bin,
        selection_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(10)])
) -> Tuple[np.ndarray, np.ndarray]:
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

    size_1 = fingerprints_1.shape[0]
    size_2 = fingerprints_2.shape[0]

    num_bins = len(selection_bins)

    selected_pairs_per_bin = -1 * np.ones((num_bins, size_1 + size_2, max_pairs_per_bin), dtype=np.int32)
    selected_scores_per_bin = np.zeros((num_bins, size_1 + size_2, max_pairs_per_bin), dtype=np.float32)

    for idx_fingerprint_i in prange(size_1):
        fingerprint_i = fingerprints_1[idx_fingerprint_i, :]
        tanimoto_scores = tanimoto_scores_row(fingerprint_i, fingerprints_2)

        for bin_number in range(num_bins):
            selection_bin = selection_bins[bin_number]
            indices = np.nonzero((tanimoto_scores > selection_bin[0]) & (tanimoto_scores <= selection_bin[1]))[0]

            np.random.shuffle(indices)
            indices = indices[:max_pairs_per_bin]
            num_indices = len(indices)
            selected_scores_per_bin[bin_number, idx_fingerprint_i, :num_indices] = tanimoto_scores[indices]
            selected_pairs_per_bin[bin_number, idx_fingerprint_i, :num_indices] = indices + size_1

    for idx_fingerprint_2 in prange(size_2):
        fingerprint_i = fingerprints_2[idx_fingerprint_2, :]
        idx_fingerprint_corrected = idx_fingerprint_2 + size_1
        tanimoto_scores = tanimoto_scores_row(fingerprint_i, fingerprints_2)

        for bin_number in range(num_bins):
            selection_bin = selection_bins[bin_number]
            indices = np.nonzero((tanimoto_scores > selection_bin[0]) & (tanimoto_scores <= selection_bin[1]))[0]

            np.random.shuffle(indices)
            indices = indices[:max_pairs_per_bin]
            num_indices = len(indices)

            selected_pairs_per_bin[bin_number, idx_fingerprint_corrected, :num_indices] = indices
            selected_scores_per_bin[bin_number, idx_fingerprint_corrected, :num_indices] = tanimoto_scores[indices]

    return selected_pairs_per_bin, selected_scores_per_bin
