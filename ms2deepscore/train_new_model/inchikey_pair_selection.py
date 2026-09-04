from collections import Counter
from typing import List, Tuple
import heapq
import numpy as np
from matchms import Spectrum
from tqdm import tqdm
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model import SpectrumPairGenerator
from ms2deepscore.fingerprint_utils import (
    derive_fingerprint_from_smiles,
    normalize_to_smiles,
)
from ms2deepscore.pair_selection_cache import (
    PairSelectionCache,
    resolve_pair_selection_cache_directory,
)
from ms2deepscore.fingerprint_similarity_computations import compute_tanimoto_similarity_per_bin


def create_spectrum_pair_generator(
        spectra: List[Spectrum],
        settings: SettingsMS2Deepscore,
        cache_directory=None,
) -> SpectrumPairGenerator:
    """Return a balanced SpectrumPairGenerator, optionally using persistent caches.

    The persistent cache stores both the expensive per-bin candidate pool and
    the final balanced pair schedule. Cache identity contains only structural
    spectrum metadata plus the settings that can influence each artifact.
    """
    if cache_directory is None:
        cache_directory = resolve_pair_selection_cache_directory(settings)
    cache = PairSelectionCache(cache_directory) if cache_directory is not None else None
    candidate_dir = None

    if cache is not None:
        cached = cache.load_candidates((spectra,), settings, mode="same_set")
    else:
        cached = None

    if cached is not None:
        inchikeys14_unique, available_pairs_per_bin_matrix, available_scores_per_bin_matrix, candidate_dir = cached
        selected_pairs = cache.load_selected_pairs(candidate_dir, settings)
        if selected_pairs is not None:
            print(f"Reusing {len(selected_pairs)} cached training compound pairs from {candidate_dir}")
            return SpectrumPairGenerator(
                selected_pairs, spectra, settings.shuffle, settings.random_seed
            )
        print(f"Reusing cached Tanimoto candidate pairs from {candidate_dir}")
    else:
        fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
            spectra,
            settings.fingerprint_type,
            settings.fingerprint_nbits,
        )

        if len(inchikeys14_unique) < settings.batch_size:
            raise ValueError("The number of unique inchikeys must be larger than the batch size.")

        max_pairs_per_bin = settings.max_pairs_per_bin
        if max_pairs_per_bin is None:
            # Honor the documented setting. This can be extremely memory hungry,
            # so bounded max_pairs_per_bin remains strongly recommended.
            max_pairs_per_bin = len(inchikeys14_unique)

        available_pairs_per_bin_matrix, available_scores_per_bin_matrix = compute_tanimoto_similarity_per_bin(
            fingerprints,
            max_pairs_per_bin,
            fingerprint_type=settings.fingerprint_type,
            selection_bins=settings.same_prob_bins,
            include_diagonal=settings.include_diagonal,
            random_seed=settings.random_seed,
        )

        if cache is not None:
            candidate_dir = cache.save_candidates(
                (spectra,),
                settings,
                mode="same_set",
                inchikeys=inchikeys14_unique,
                available_pairs=available_pairs_per_bin_matrix,
                available_scores=available_scores_per_bin_matrix,
            )
            # Re-open as memory maps so the rest of the workflow does not need
            # another full in-memory copy of the cached arrays.
            cached = cache.load_candidates((spectra,), settings, mode="same_set")
            if cached is not None:
                inchikeys14_unique, available_pairs_per_bin_matrix, available_scores_per_bin_matrix, candidate_dir = cached

    if len(inchikeys14_unique) < settings.batch_size:
        raise ValueError("The number of unique inchikeys must be larger than the batch size.")

    pair_frequency_matrixes = balanced_selection_of_pairs_per_bin(
        available_pairs_per_bin_matrix, settings
    )

    selected_pairs_per_bin = convert_to_selected_pairs_list(
        pair_frequency_matrixes,
        available_pairs_per_bin_matrix,
        available_scores_per_bin_matrix,
        inchikeys14_unique,
    )
    selected_pairs = [pair for pairs in selected_pairs_per_bin for pair in pairs]

    if cache is not None and candidate_dir is not None:
        cache.save_selected_pairs(candidate_dir, settings, selected_pairs)

    return SpectrumPairGenerator(
        selected_pairs, spectra, settings.shuffle, settings.random_seed
    )


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

        # Normalize InChI before appending the matching inchikey.
        normalized_structure = normalize_to_smiles(structure)
        if normalized_structure is None:
            continue

        structure_list.append(normalized_structure)
        valid_inchikeys.append(inchikey14)

    if len(structure_list) == 0:
        raise ValueError("No valid SMILES/InChI entries available for fingerprint calculation")

    fingerprints = derive_fingerprint_from_smiles(
        structure_list,
        fingerprint_type=fingerprint_type,
        nbits=nbits,
        policy_invalid_smiles="keep",
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
    """Convert pair frequencies to ``(inchikey1, inchikey2, score)`` lists.

    The previous implementation (version<=0.29) iterated in Python over every slot of the
    dense ``(bins, compounds, max_pairs_per_bin)`` candidate cube. At large
    scale that can mean hundreds of millions of Python-loop iterations even
    though only a small fraction of slots have a non-zero selected frequency.
    Here NumPy finds non-zero entries in C and canonical pair IDs remove the
    mirrored duplicates before the much smaller Python expansion step.

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
    nr_of_inchikeys = len(inchikeys14_unique)

    for bin_id in tqdm(range(pair_frequency_matrixes.shape[0])):
        frequency_matrix = pair_frequency_matrixes[bin_id]
        row_indices, column_indices = np.nonzero(frequency_matrix > 0)

        if len(row_indices) == 0:
            selected_pairs_per_bin.append([])
            continue

        partner_indices = available_pairs_per_bin_matrix[
            bin_id, row_indices, column_indices
        ]
        valid = partner_indices >= 0
        row_indices = row_indices[valid]
        column_indices = column_indices[valid]
        partner_indices = partner_indices[valid]

        lower = np.minimum(row_indices, partner_indices).astype(np.int64)
        upper = np.maximum(row_indices, partner_indices).astype(np.int64)
        pair_codes = lower * nr_of_inchikeys + upper

        # ``np.unique`` sorts by code; sort the returned first-occurrence
        # positions to retain the legacy row-major ordering as closely as
        # possible while dropping mirrored duplicates.
        _, first_occurrences = np.unique(pair_codes, return_index=True)
        first_occurrences.sort()

        selected_pairs = []
        for position in first_occurrences:
            inchikey1_index = int(lower[position])
            inchikey2_index = int(upper[position])
            column_index = int(column_indices[position])
            original_row = int(row_indices[position])
            pair_frequency = int(frequency_matrix[original_row, column_index])
            score = float(scores_matrix[bin_id, original_row, column_index])

            selected_pairs.extend(
                [(
                    inchikeys14_unique[inchikey1_index],
                    inchikeys14_unique[inchikey2_index],
                    score,
                )] * pair_frequency
            )

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

    # Initialize pair frequencies with the smallest safe integer dtype.
    sentinel_value = 2 * max_resampling
    frequency_dtype = (
        np.int32
        if sentinel_value <= np.iinfo(np.int32).max
        else np.int64
    )
    pair_frequency = np.zeros_like(
        available_pairs_for_bin_matrix, dtype=frequency_dtype
    )

    # Mask for invalid pairs (where value is -1)
    invalid_mask = (available_pairs_for_bin_matrix == -1)
    pair_frequency[invalid_mask] = sentinel_value  # Ensure these pairs are never selected

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

            # Among valid pairs, find those with the lowest pair frequency.
            # Keep the validity mask when resolving ties: the previous code
            # could re-introduce partners that were already above
            # max_inchikey_count merely because they had the same pair count.
            candidate_mask = valid_pairs_mask & valid_inchikeys_mask
            min_pair_freq = np.min(pair_freq_row[candidate_mask])
            min_freq_mask = candidate_mask & (pair_freq_row == min_pair_freq)

            # From the least-resampled pairs select the least-sampled partner.
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

        spectra_selected.append(list_of_spectra[ID])

    return spectra_selected, inchikeys14_unique
