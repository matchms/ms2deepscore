import json
from typing import List, Tuple
from collections import Counter
import numpy as np
from matchms import Spectrum
from numba import jit, prange
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.TrainingBatchGenerator import TrainingBatchGenerator
from ms2deepscore.train_new_model.SpectrumPairGenerator import SpectrumPairGenerator
from ms2deepscore.train_new_model.inchikey_pair_selection import compute_fingerprints_for_training, \
    balanced_selection_of_pairs_per_bin, convert_to_selected_pairs_list, tanimoto_scores_row, \
    create_spectrum_pair_generator
from ms2deepscore.utils import split_by_ionmode

def create_data_generator_across_ionmodes(training_spectra,
                                          settings: SettingsMS2Deepscore) -> TrainingBatchGenerator:
    pos_spectra, neg_spectra = split_by_ionmode(training_spectra)

    pos_spectrum_pair_generator = create_spectrum_pair_generator(pos_spectra, settings=settings)
    neg_spectrum_pair_generator = create_spectrum_pair_generator(neg_spectra, settings=settings)
    pos_neg_spectrum_pair_generator = select_compound_pairs_wrapper_across_ionmode(pos_spectra, neg_spectra, settings)

    spectrum_pair_generator = CombinedSpectrumGenerator([pos_spectrum_pair_generator, neg_spectrum_pair_generator, pos_neg_spectrum_pair_generator])

    train_generator = TrainingBatchGenerator(spectrum_pair_generator=spectrum_pair_generator, settings=settings)
    return train_generator


def select_compound_pairs_wrapper_across_ionmode(
        spectra_1: List[Spectrum],
        spectra_2: List[Spectrum],
        settings: SettingsMS2Deepscore,
) -> "SpectrumPairGeneratorAcrossIonmodes":
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
    return SpectrumPairGeneratorAcrossIonmodes([pair for pairs in selected_pairs_per_bin for pair in pairs],
                                               spectra_1, spectra_2, settings.shuffle, settings.random_seed)


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


class SpectrumPairGeneratorAcrossIonmodes:
    def __init__(self, selected_inchikey_pairs: List[Tuple[str, str, float]],
                 spectra_pos: List[Spectrum], spectra_neg: List[Spectrum],
                 shuffle: bool = True, random_seed: int = 0):
        """
        Parameters
        ----------
        selected_inchikey_pairs:
            A list with tuples encoding inchikey pairs like: (inchikey1, inchikey2, tanimoto_score)
        """
        self.selected_inchikey_pairs = selected_inchikey_pairs
        self.spectra_pos = spectra_pos
        self.spectra_neg = spectra_neg

        self.pos_inchikeys = np.array([s.get("inchikey")[:14] for s in self.spectra_pos])
        self.neg_inchikeys= np.array([s.get("inchikey")[:14] for s in self.spectra_neg])

        self.shuffle = shuffle
        self.random_nr_generator = np.random.default_rng(random_seed)
        self._idx = 0
        if self.shuffle:
            self.random_nr_generator.shuffle(self.selected_inchikey_pairs)

    def __iter__(self):
        return self

    def __next__(self):
        # reshuffle when we've gone through everything
        if self._idx >= len(self.selected_inchikey_pairs):
            self._idx = 0
            if self.shuffle:
                self.random_nr_generator.shuffle(self.selected_inchikey_pairs)

        inchikey1, inchikey2, tanimoto_score = self.selected_inchikey_pairs[self._idx]
        spectrum1 = self._get_pos_spectrum_with_inchikey(inchikey1, self.random_nr_generator)
        spectrum2 = self._get_neg_spectrum_with_inchikey(inchikey2, self.random_nr_generator)
        self._idx += 1
        return spectrum1, spectrum2, tanimoto_score

    def __len__(self):
        return len(self.selected_inchikey_pairs)

    def __str__(self):
        return f"SpectrumPairGenerator with {len(self.selected_inchikey_pairs)} pairs available"

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

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data_for_json, f)

    def _get_pos_spectrum_with_inchikey(self, inchikey: str, random_number_generator) -> Spectrum:
        matching_spectrum_id = np.where(self.pos_inchikeys == inchikey)[0]
        if len(matching_spectrum_id) <= 0:
            raise ValueError("No matching inchikey found (note: expected first 14 characters), "
                             "likely switched pos and neg in entry")
        return self.spectra_pos[random_number_generator.choice(matching_spectrum_id)]

    def _get_neg_spectrum_with_inchikey(self, inchikey: str, random_number_generator) -> Spectrum:
        matching_spectrum_id = np.where(self.neg_inchikeys == inchikey)[0]
        if len(matching_spectrum_id) <= 0:
            raise ValueError("No matching inchikey found (note: expected first 14 characters), "
                             "likely switched pos and neg in entry")
        return self.spectra_neg[random_number_generator.choice(matching_spectrum_id)]


class CombinedSpectrumGenerator:
    """Combines multiple SpectrumPairGenerators into a single generator

    This is used to combine different iterators for each ionmode pair"""
    def __init__(self, spectrum_pair_generators: List[SpectrumPairGenerator]):
        self.generators = spectrum_pair_generators
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.generators:
            raise StopIteration
        current_generator = self.generators[self._idx % len(self.generators)]
        self._idx += 1
        return next(current_generator)

    def __len__(self):
        return sum([len(generator) for generator in self.generators])