from collections import Counter
from typing import List, Tuple
import heapq
import numpy as np
from matchms import Spectrum
from tqdm import tqdm
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model import SpectrumPairGenerator
from ms2deepscore.fingerprint_utils import derive_fingerprint_from_smiles_or_inchi
from ms2deepscore.fingerprint_similarity_computations import compute_tanimoto_similarity_per_bin


def create_spectrum_pair_generator(
        spectra: List[Spectrum],
        settings: SettingsMS2Deepscore,
) -> SpectrumPairGenerator:
    """Returns a SpectrumPairGenerator object containing equally balanced pairs over the different bins

    spectra:
        A list of spectra
    settings:
        The settings that should be used for selecting the compound pairs wrapper. The settings should be specified as a
        SettingsMS2Deepscore object.

    Returns
    -------
    SpectrumPairGenerator
        SpectrumPairGenerator containing balanced pairs. The pairs are stored as [(inchikey1, inchikey2, score)]
    """
    if settings.random_seed is not None:
        np.random.seed(settings.random_seed)

    fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
        spectra,
        settings.fingerprint_type,
        settings.fingerprint_nbits
        )

    if len(inchikeys14_unique) < settings.batch_size:
        raise ValueError("The number of unique inchikeys must be larger than the batch size.")

    available_pairs_per_bin_matrix, available_scores_per_bin_matrix = compute_tanimoto_similarity_per_bin(
        fingerprints,
        settings.max_pairs_per_bin,
        fingerprint_type=settings.fingerprint_type,
        selection_bins=settings.same_prob_bins,
        include_diagonal=settings.include_diagonal,
    )
    pair_frequency_matrixes = balanced_selection_of_pairs_per_bin(
        available_pairs_per_bin_matrix, settings)

    selected_pairs_per_bin = convert_to_selected_pairs_list(
        pair_frequency_matrixes, available_pairs_per_bin_matrix,
        available_scores_per_bin_matrix, inchikeys14_unique)

    return SpectrumPairGenerator([pair for pairs in selected_pairs_per_bin for pair in pairs],
                                 spectra, settings.shuffle, settings.random_seed)


def compute_fingerprints_for_training(
    spectra: List[Spectrum],
    fingerprint_type: str = "rdkit_binary",
    nbits: int = 2048
) -> Tuple[np.ndarray, List[str]]:
    """Calculates fingerprints for each unique inchikey.

    Function returns only the inchikeys for which a fingerprint could be calculated.

    Parameters
    ----------
    spectra:
        The spectra for which fingerprints should be calculated
    fingerprint_type:
        The fingerprint type that should be used for tanimoto score calculations.
    nbits:
        The number of bits to use for the fingerprint.
    """
    if len(spectra) == 0:
        raise ValueError("No spectra were selected to calculate fingerprints")

    spectra_selected, inchikeys14_unique = select_inchi_for_unique_inchikeys(spectra)
    print(f"Selected {len(spectra_selected)} spectra with unique inchikeys for calculating tanimoto scores "
          f"(out of {len(spectra)} spectra)")

    if len(spectra_selected) == 0:
        raise ValueError("No spectra with valid structural annotations were found")

    structure_list = []
    valid_inchikeys = []

    for spectrum, inchikey14 in zip(spectra_selected, inchikeys14_unique):
        structure = spectrum.get("smiles")
        if structure is None:
            structure = spectrum.get("inchi")

        if structure is None:
            continue

        structure_list.append(structure)
        valid_inchikeys.append(inchikey14)

    if len(structure_list) == 0:
        raise ValueError("No valid SMILES/InChI entries available for fingerprint calculation")

    fingerprints = derive_fingerprint_from_smiles_or_inchi(
        structure_list,
        fingerprint_type=fingerprint_type,
        nbits=nbits,
        policy_invalid="keep",
    )

    if len(fingerprints) == 0:
        raise ValueError("No fingerprints could be computed")

    if len(valid_inchikeys) != len(fingerprints):
        raise ValueError(
            f"Mismatch between inchikeys ({len(valid_inchikeys)}) and fingerprints ({len(fingerprints)})."
        )

    return fingerprints, valid_inchikeys


def determine_nr_of_pairs_per_bin(settings, nr_of_inchikeys):
    """Calculate the target number of pairs per bin based on nr of unique inchikeys and given settings.

    Parameters:
    -----------
    settings:
        Settings object containing configuration options. 
        Required attributes:
            - average_inchikey_sampling_count: The desired average number of inchikeys selected
            - same_prob_bins: The probability bins used
    nr_of_inchikeys:
        The total number of InChIKeys.
    """

    # Calculate initial target number of pairs per bin
    average_inchikey_sampling_per_bin = settings.average_inchikey_sampling_count/len(settings.same_prob_bins)
    nr_of_inchikeys_sampled_per_bin = average_inchikey_sampling_per_bin * nr_of_inchikeys
    aimed_nr_of_pairs_per_bin = int(nr_of_inchikeys_sampled_per_bin / 2)  # Each pair consists of 2 inchikeys
    return aimed_nr_of_pairs_per_bin


def balanced_selection_of_pairs_per_bin(
        available_pairs_per_bin_matrix: np.ndarray,
        settings: SettingsMS2Deepscore,
        ) -> np.ndarray:
    """From the available_pairs_per_bin_matrix a balanced selection is made to have a balanced distribution.

    The algorithm is designed to have a perfect balance over the tanimoto bins,
    a close to equal sampling of all inchikeys
    and a  close to equal distribution of pairs per inchikey over the bins.

    This is achieved by storing the inchikey counts in the sampled pairs.
    The bins are sampled in the order they appear in available_pairs_per_bin_matrix
    (which is determined by the order in settings.same_prob_bins.
    The least frequent sampled inchikeys are always sampled first,
    resulting in a well balanced distribution over the bins.

    Parameters
    ----------
    available_pairs_per_bin_matrix:
        A numpy 3D matrix. The first dimension is the tanimoto bins.
        For each tanimoto bin a matrix is stored with pairs. The indexes of the rows are the indexes of the first
        inchikey of the pair and the value given in the rows are the indexes of the second inchikey of the pair.
        If the value is -1 it indicates that there were no more pairs available for this inchikey in this bin.
    settings:
        A SettingsMS2Deepscore object
    """

    inchikey_count = np.zeros(available_pairs_per_bin_matrix.shape[1])
    nr_of_pairs_per_bin = determine_nr_of_pairs_per_bin(settings, nr_of_inchikeys=len(inchikey_count))

    pair_frequency_matrixes = []
    for pairs_in_bin in available_pairs_per_bin_matrix:
        pair_frequencies, inchikey_count = select_balanced_pairs(
            pairs_in_bin,
            inchikey_count,
            nr_of_pairs_per_bin,
            settings.max_pair_resampling,
            settings.max_inchikey_sampling)
        pair_frequency_matrixes.append(pair_frequencies)

    pair_frequency_matrixes = np.array(pair_frequency_matrixes)
    pair_frequency_matrixes[pair_frequency_matrixes == 2 * settings.max_pair_resampling] = 0
    return pair_frequency_matrixes


def convert_to_selected_pairs_list(pair_frequency_matrixes: np.ndarray,
                                   available_pairs_per_bin_matrix: np.ndarray,
                                   scores_matrix: np.ndarray,
                                   inchikeys14_unique: List[str]):
    """Convert the matrixes denoting the pairs to a list of pairs, encoded as [(inchikey1, inchikey2, score)]

    Parameters
    ----------
    pair_frequency_matrixes:
        The frequency each pair should be sampled.
        The positions correspond to available_pairs_per_bin_matrix,
        but contain the frequency of sampling for the corresponding pairs.
    available_pairs_per_bin_matrix:
        A numpy 3D matrix. The first dimension is the tanimoto bins.
        For each tanimoto bin a matrix is stored with pairs. The indexes of the rows are the indexes of the first
        inchikey of the pair and the value given in the rows are the indexes of the second inchikey of the pair.
        If the value is -1 it indicates that there were no more pairs available for this inchikey in this bin.
    scores_matrix:
        A numpy 3D matrix containing the scores per pair.
        The positions correspond to available_pairs_per_bin_matrix, but contain the scores for the corresponding pairs.
    inchikeys14_unique:
        List of inchikeys.
        This is used to map the indexes of inchikeys used in the matrixes, to the corresponding inchikeys.
    """
    selected_pairs_per_bin = []
    for bin_id, bin_pair_frequency_matrix in enumerate(tqdm(pair_frequency_matrixes)):
        selected_pairs = []
        for inchikey1_index, pair_frequency_row in enumerate(bin_pair_frequency_matrix):
            for column_index, pair_frequency in enumerate(pair_frequency_row):
                if pair_frequency > 0:
                    inchikey2_index = available_pairs_per_bin_matrix[bin_id][inchikey1_index][column_index]
                    score = scores_matrix[bin_id][inchikey1_index][column_index]
                    # This ensures that the order is the same.
                    # This is important for the cross ionization mode selection.
                    if inchikey1_index < inchikey2_index:
                        selected_pairs.extend(
                            [(inchikeys14_unique[inchikey1_index], inchikeys14_unique[inchikey2_index], score)] * pair_frequency)
                    else:
                        selected_pairs.extend(
                            [(inchikeys14_unique[inchikey2_index], inchikeys14_unique[inchikey1_index], score)] * pair_frequency)
                    # remove duplicate pairs
                    position_of_first_inchikey_in_matrix = available_pairs_per_bin_matrix[bin_id][
                                                               inchikey2_index] == inchikey1_index
                    bin_pair_frequency_matrix[inchikey2_index][position_of_first_inchikey_in_matrix] = 0
        selected_pairs_per_bin.append(selected_pairs)
    return selected_pairs_per_bin


def select_balanced_pairs(available_pairs_for_bin_matrix: np.ndarray,
                          inchikey_counts: np.ndarray,
                          required_number_of_pairs: int,
                          max_resampling: int,
                          max_inchikey_count: int):
    """Determines how frequently each available pair should be sampled.

    Inchikey pairs are selected by first selecting the least frequent inchikey. For this inchikey, all available pairs
    are considered. The pair is picked where the second inchikey has the lowest frequency in inchikey_counts.

    Parameters
    ----------
    available_pairs_for_bin_matrix:
        A 2D numpy array storing the available inchikey pairs for the current bin. The rows represent the first inchikey
        of the pair, and the values in the rows are the indexes of the second inchikey of the pair.
        A value of -1 indicates no more pairs are available for this inchikey in this bin.
    inchikey_counts:
        An array representing the number of times each inchikey has been sampled. This is used to determine which pairs
        should be sampled first. The inchikey counts as input already contain the counts from previous bins.
    max_resampling:
        The maximum number of times a pair can be resampled.
        Resampling means that the exact same inchikey pair is added multiple times to the list of pairs.
    max_inchikey_count:
        The number of pairs to sample.

    Returns
    -------
    pair_frequency:
        A 2D array matching available_pairs_for_bin_matrix in dimensions. Each position encodes the number of times the
        corresponding pair should be sampled.
    inchikey_counts:
        The updated inchikey counts.
    """
    num_inchikeys = available_pairs_for_bin_matrix.shape[0]

    # Initialize pair frequency matrix
    pair_frequency = np.zeros_like(available_pairs_for_bin_matrix, dtype=int)

    # Mask for invalid pairs (where value is -1)
    invalid_mask = (available_pairs_for_bin_matrix == -1)
    pair_frequency[invalid_mask] = max_resampling * 2  # Ensure these pairs are never selected

    # Initialize available inchikeys as a min-heap based on inchikey_counts
    available_inchikey_indexes = [(inchikey_counts[i], i) for i in range(num_inchikeys)
                                  if not np.all(pair_frequency[i] >= max_resampling)]
    heapq.heapify(available_inchikey_indexes)

    nr_of_pairs_selected = 0

    with tqdm(total=required_number_of_pairs,
              desc="Balanced sampling of inchikey pairs (per bin)") as progress_bar:
        while nr_of_pairs_selected < required_number_of_pairs:
            if not available_inchikey_indexes:
                raise ValueError("The number of pairs available is less than required_number_of_pairs. "
                                 f"Only {nr_of_pairs_selected} pairs could be selected in this bin, "
                                 f"but {required_number_of_pairs} pairs are required. "
                                 "Increase max_pair_resampling or decrease average_inchikey_sampling_count.")

            # Pop the inchikey with the lowest count
            _, inchikey_with_lowest_count = heapq.heappop(available_inchikey_indexes)

            if inchikey_counts[inchikey_with_lowest_count] > max_inchikey_count:
                raise ValueError("There are not enough inchikeys with a pair in the current bin "
                                 "that have less than max_inchikey_count"
                                 f"Only {nr_of_pairs_selected} pairs could be selected in this bin, "
                                 f"but {required_number_of_pairs} pairs are required"
                                 "Increase max_inchikey_count or decrease average_inchikey_sampling_count")

            # Get pair frequencies
            pair_freq_row = pair_frequency[inchikey_with_lowest_count]

            # Get available second inchikeys
            available_pairs_row = available_pairs_for_bin_matrix[inchikey_with_lowest_count]

            # Get counts for second inchikeys
            second_inchikey_counts = inchikey_counts[available_pairs_row]

            # Select inchikeys that have an inchikey count below max_inchikey_count
            valid_inchikeys_mask = second_inchikey_counts < max_inchikey_count

            # Find inchikey indices where pair frequency is less than max_resampling
            valid_pairs_mask = pair_freq_row < max_resampling

            if not np.any(valid_pairs_mask & valid_inchikeys_mask):
                continue  # No valid pairs left for this inchikey

            # Among valid pairs, find those with the lowest pair frequency
            min_pair_freq = np.min(pair_freq_row[valid_pairs_mask & valid_inchikeys_mask])
            min_freq_mask = pair_freq_row == min_pair_freq

            # From the least resampled inchikey select the leas sampled inchikey
            min_inchikey_count_idx = np.argmin(second_inchikey_counts[min_freq_mask])
            second_inchikey_with_lowest_count = available_pairs_row[min_freq_mask][min_inchikey_count_idx]

            # Update pair frequency
            pair_indices = np.where(available_pairs_row == second_inchikey_with_lowest_count)[0]
            pair_frequency[inchikey_with_lowest_count, pair_indices] += 1

            # If the pair is not symmetrical, update the reverse pair frequency
            if second_inchikey_with_lowest_count != inchikey_with_lowest_count:
                reverse_pairs_row = available_pairs_for_bin_matrix[second_inchikey_with_lowest_count]
                reverse_pair_indices = np.where(reverse_pairs_row == inchikey_with_lowest_count)[0]
                pair_frequency[second_inchikey_with_lowest_count, reverse_pair_indices] += 1

            # Update inchikey counts
            inchikey_counts[inchikey_with_lowest_count] += 1
            inchikey_counts[second_inchikey_with_lowest_count] += 1

            nr_of_pairs_selected += 1
            progress_bar.update(1)

            # If this inchikey still has valid pairs, push it back into the heap
            if np.any(pair_frequency[inchikey_with_lowest_count] < max_resampling):
                heapq.heappush(available_inchikey_indexes,
                               (inchikey_counts[inchikey_with_lowest_count], inchikey_with_lowest_count))

    return pair_frequency, inchikey_counts


def get_nr_of_available_pairs_in_bin(selected_pairs_per_bin_matrix: np.ndarray) -> List[int]:
    """Calculates the number of unique pairs available per bin, discarding duplicated (inverted) pairs.
    """
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


def select_inchi_for_unique_inchikeys(
        list_of_spectra: List['Spectrum']
) -> Tuple[List['Spectrum'], List[str]]:
    """Select spectra with most frequent inchi for unique inchikeys.

    Method needed to calculate Tanimoto scores.
    """
    inchikeys_list = [s.get("inchikey") for s in list_of_spectra]
    inchi_list = [s.get("inchi") for s in list_of_spectra]

    inchi_array = np.array(inchi_list)
    inchikeys14_array = np.array([x[:14] for x in inchikeys_list])

    inchikeys14_unique = sorted(set(inchikeys14_array))

    spectra_selected = []
    for inchikey14 in inchikeys14_unique:
        idx = np.where(inchikeys14_array == inchikey14)[0]

        most_common_inchi = Counter(inchi_array[idx]).most_common(1)[0][0]
        # ID of the spectrum with the most frequent inchi
        ID = idx[np.where(inchi_array[idx] == most_common_inchi)[0][0]]

        spectra_selected.append(list_of_spectra[ID].clone())

    return spectra_selected, inchikeys14_unique
