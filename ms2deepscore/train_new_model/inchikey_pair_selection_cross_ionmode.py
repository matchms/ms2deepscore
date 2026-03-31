import json
from typing import List, Tuple
from collections import Counter

import numpy as np
from matchms import Spectrum

from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.TrainingBatchGenerator import TrainingBatchGenerator
from ms2deepscore.train_new_model.SpectrumPairGenerator import SpectrumPairGenerator
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    compute_fingerprints_for_training,
    balanced_selection_of_pairs_per_bin,
    convert_to_selected_pairs_list,
    create_spectrum_pair_generator,
)
from ms2deepscore.fingerprint_similarity_computations import (
    compute_tanimoto_similarity_per_bin_between_sets,
)
from ms2deepscore.utils import split_by_ionmode


def create_data_generator_across_ionmodes(
    training_spectra,
    settings: SettingsMS2Deepscore,
) -> TrainingBatchGenerator:
    pos_spectra, neg_spectra = split_by_ionmode(training_spectra)

    pos_spectrum_pair_generator = create_spectrum_pair_generator(pos_spectra, settings=settings)
    neg_spectrum_pair_generator = create_spectrum_pair_generator(neg_spectra, settings=settings)
    pos_neg_spectrum_pair_generator = select_compound_pairs_wrapper_across_ionmode(
        pos_spectra, neg_spectra, settings
    )

    spectrum_pair_generator = CombinedSpectrumGenerator(
        [pos_spectrum_pair_generator, neg_spectrum_pair_generator, pos_neg_spectrum_pair_generator]
    )

    train_generator = TrainingBatchGenerator(
        spectrum_pair_generator=spectrum_pair_generator, settings=settings
    )
    return train_generator


def select_compound_pairs_wrapper_across_ionmode(
        spectra_1: List[Spectrum],
        spectra_2: List[Spectrum],
        settings: SettingsMS2Deepscore,
) -> "SpectrumPairGeneratorAcrossIonmodes":
    """Returns a SpectrumPairGeneratorAcrossIonmodes object containing equally balanced cross-ionmode pairs.

    Parameters
    ----------
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
        settings.fingerprint_nbits,
    )
    fingerprints_2, inchikeys14_unique_2 = compute_fingerprints_for_training(
        spectra_2,
        settings.fingerprint_type,
        settings.fingerprint_nbits,
    )

    if len(inchikeys14_unique_1) < settings.batch_size or len(inchikeys14_unique_2) < settings.batch_size:
        raise ValueError("The number of unique inchikeys must be larger than the batch size.")

    available_pairs_per_bin_matrix, available_scores_per_bin_matrix = (
        compute_tanimoto_similarity_per_bin_between_sets(
            fingerprints_1,
            fingerprints_2,
            max_pairs_per_bin=settings.max_pairs_per_bin,
            fingerprint_type=settings.fingerprint_type,
            selection_bins=settings.same_prob_bins,
        )
    )

    pair_frequency_matrixes = balanced_selection_of_pairs_per_bin(
        available_pairs_per_bin_matrix, settings
    )

    selected_pairs_per_bin = convert_to_selected_pairs_list(
        pair_frequency_matrixes,
        available_pairs_per_bin_matrix,
        available_scores_per_bin_matrix,
        inchikeys14_unique_1 + inchikeys14_unique_2,
    )

    return SpectrumPairGeneratorAcrossIonmodes(
        [pair for pairs in selected_pairs_per_bin for pair in pairs],
        spectra_1,
        spectra_2,
        settings.shuffle,
        settings.random_seed,
    )


class SpectrumPairGeneratorAcrossIonmodes:
    def __init__(
        self,
        selected_inchikey_pairs: List[Tuple[str, str, float]],
        spectra_pos: List[Spectrum],
        spectra_neg: List[Spectrum],
        shuffle: bool = True,
        random_seed: int = 0,
    ):
        self.selected_inchikey_pairs = selected_inchikey_pairs
        self.spectra_pos = spectra_pos
        self.spectra_neg = spectra_neg

        self.pos_inchikeys = np.array([s.get("inchikey")[:14] for s in self.spectra_pos])
        self.neg_inchikeys = np.array([s.get("inchikey")[:14] for s in self.spectra_neg])

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
                inchikey_scores[inchikey_1] = [score]

            if inchikey_2 in inchikey_scores:
                inchikey_scores[inchikey_2].append(score)
            else:
                inchikey_scores[inchikey_2] = [score]
        return inchikey_scores

    def save_as_json(self, file_name):
        data_for_json = [(item[0], item[1], float(item[2])) for item in self.selected_inchikey_pairs]
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data_for_json, f)

    def _get_pos_spectrum_with_inchikey(self, inchikey: str, random_number_generator) -> Spectrum:
        matching_spectrum_id = np.where(self.pos_inchikeys == inchikey)[0]
        if len(matching_spectrum_id) <= 0:
            raise ValueError(
                "No matching inchikey found (note: expected first 14 characters), likely switched pos and neg in entry"
            )
        return self.spectra_pos[random_number_generator.choice(matching_spectrum_id)]

    def _get_neg_spectrum_with_inchikey(self, inchikey: str, random_number_generator) -> Spectrum:
        matching_spectrum_id = np.where(self.neg_inchikeys == inchikey)[0]
        if len(matching_spectrum_id) <= 0:
            raise ValueError(
                "No matching inchikey found (note: expected first 14 characters), likely switched pos and neg in entry"
            )
        return self.spectra_neg[random_number_generator.choice(matching_spectrum_id)]


class CombinedSpectrumGenerator:
    """Combines multiple SpectrumPairGenerators into a single generator."""

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
        return sum(len(generator) for generator in self.generators)
