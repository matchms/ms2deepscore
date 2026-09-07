import json
from pathlib import Path
from typing import List, Tuple, Union
from collections import Counter
from collections import defaultdict
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
from ms2deepscore.train_new_model.pair_data_persistence import (
    SelectedPairSchedule,
    build_pair_data_manifest,
    candidate_data_exists,
    load_candidate_pair_data,
    load_selected_pair_schedule,
    prepare_pair_data_folder,
    save_candidate_pair_data,
    save_selected_pair_schedule,
    selected_schedule_exists,
    spectra_structure_signature,
)


def create_data_generator_across_ionmodes(
    training_spectra,
    settings: SettingsMS2Deepscore,
    pair_data_folder: Union[str, Path, None] = None,
) -> TrainingBatchGenerator:
    """Create balanced positive, negative and cross-ionmode training generators.

    When ``pair_data_folder`` is given, explicit subfolders ``positive``,
    ``negative`` and ``positive_negative`` are used for persisted pair data.
    """
    pos_spectra, neg_spectra = split_by_ionmode(training_spectra)

    base_folder = Path(pair_data_folder) if pair_data_folder is not None else None
    if base_folder is not None:
        # A top-level manifest makes the folder layout explicit and prevents accidentally
        # reusing a folder that was previously prepared for single/within-set training.
        layout_manifest = build_pair_data_manifest(
            kind="balanced_across_ionmodes",
            settings=settings,
            spectra_signature_1=spectra_structure_signature(training_spectra),
        )
        prepare_pair_data_folder(base_folder, layout_manifest)

    pos_folder = base_folder / "positive" if base_folder is not None else None
    neg_folder = base_folder / "negative" if base_folder is not None else None
    cross_folder = base_folder / "positive_negative" if base_folder is not None else None

    pos_spectrum_pair_generator = create_spectrum_pair_generator(
        pos_spectra, settings=settings, pair_data_folder=pos_folder
    )
    neg_spectrum_pair_generator = create_spectrum_pair_generator(
        neg_spectra, settings=settings, pair_data_folder=neg_folder
    )
    pos_neg_spectrum_pair_generator = select_compound_pairs_wrapper_across_ionmode(
        pos_spectra, neg_spectra, settings, pair_data_folder=cross_folder
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
        pair_data_folder: Union[str, Path, None] = None,
) -> "SpectrumPairGeneratorAcrossIonmodes":
    """Create cross-ionmode pairs, optionally using explicit persisted pair data.
        Parameters
    ----------
    spectra_1:
        A list of spectra
    spectra_2:
        A list of spectra
    settings:
        The settings that should be used for selecting the compound pairs wrapper. The settings should be specified as a
        SettingsMS2Deepscore object.
    pair_data_folder:
        The folder where the pair data should be stored. If None, the pair data will not be stored.
    """
    if settings.random_seed is not None:
        np.random.seed(settings.random_seed)

    pair_folder = None
    if pair_data_folder is not None:
        manifest = build_pair_data_manifest(
            kind="between_sets",
            settings=settings,
            spectra_signature_1=spectra_structure_signature(spectra_1),
            spectra_signature_2=spectra_structure_signature(spectra_2),
        )
        pair_folder = prepare_pair_data_folder(pair_data_folder, manifest)
        candidate_available = candidate_data_exists(pair_folder)
        if selected_schedule_exists(pair_folder):
            print(f"Loading selected cross-ionmode pair schedule from {pair_folder}")
            schedule = load_selected_pair_schedule(pair_folder)
            return SpectrumPairGeneratorAcrossIonmodes(
                schedule, spectra_1, spectra_2, settings.shuffle, settings.random_seed
            )

    if pair_folder is not None and candidate_available:
        print(f"Loading cross-ionmode candidate pair/score data from {pair_folder}")
        available_pairs_per_bin_matrix, available_scores_per_bin_matrix, inchikeys14_unique = (
            load_candidate_pair_data(pair_folder)
        )
    else:
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
        inchikeys14_unique = inchikeys14_unique_1 + inchikeys14_unique_2
        if pair_folder is not None:
            print(f"Saving cross-ionmode candidate pair/score data to {pair_folder}")
            save_candidate_pair_data(
                pair_folder,
                available_pairs_per_bin_matrix,
                available_scores_per_bin_matrix,
                inchikeys14_unique,
            )

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

    if pair_folder is not None:
        schedule = SelectedPairSchedule.from_pairs(selected_pairs, inchikeys14_unique)
        print(f"Saving selected cross-ionmode pair schedule to {pair_folder}")
        save_selected_pair_schedule(pair_folder, schedule)
        selected_pairs_for_generator = schedule
    else:
        selected_pairs_for_generator = selected_pairs

    return SpectrumPairGeneratorAcrossIonmodes(
        selected_pairs_for_generator,
        spectra_1,
        spectra_2,
        settings.shuffle,
        settings.random_seed,
    )


class SpectrumPairGeneratorAcrossIonmodes:
    def __init__(
        self,
        selected_inchikey_pairs: Union[List[Tuple[str, str, float]], SelectedPairSchedule],
        spectra_pos: List[Spectrum],
        spectra_neg: List[Spectrum],
        shuffle: bool = True,
        random_seed: int = 0,
    ):
        self.selected_inchikey_pairs = selected_inchikey_pairs
        self.spectra_pos = spectra_pos
        self.spectra_neg = spectra_neg

        self.pos_spectra_by_inchikey = self._build_spectra_by_inchikey(self.spectra_pos)
        self.neg_spectra_by_inchikey = self._build_spectra_by_inchikey(self.spectra_neg)

        self.shuffle = shuffle
        self.random_nr_generator = np.random.default_rng(random_seed)
        self._idx = 0
        self._compact_schedule = isinstance(self.selected_inchikey_pairs, SelectedPairSchedule)
        if self._compact_schedule:
            self._pair_order = np.arange(len(self.selected_inchikey_pairs), dtype=np.int64)
            if self.shuffle:
                self.random_nr_generator.shuffle(self._pair_order)
        elif self.shuffle:
            self.random_nr_generator.shuffle(self.selected_inchikey_pairs)

    @staticmethod
    def _build_spectra_by_inchikey(spectra: List[Spectrum]) -> dict[str, np.ndarray]:
        """Create fast lookup from inchikey14 to spectrum indices."""
        spectra_by_inchikey = defaultdict(list)

        for spectrum_id, spectrum in enumerate(spectra):
            inchikey = spectrum.get("inchikey")
            if inchikey is None:
                continue
            spectra_by_inchikey[inchikey[:14]].append(spectrum_id)

        return {
            inchikey: np.asarray(indices, dtype=np.int64)
            for inchikey, indices in spectra_by_inchikey.items()
        }

    def __iter__(self):
        return self

    def __next__(self):
        # reshuffle when we've gone through everything
        if self._idx >= len(self.selected_inchikey_pairs):
            self._idx = 0
            if self.shuffle:
                if self._compact_schedule:
                    self.random_nr_generator.shuffle(self._pair_order)
                else:
                    self.random_nr_generator.shuffle(self.selected_inchikey_pairs)

        pair_index = int(self._pair_order[self._idx]) if self._compact_schedule else self._idx
        inchikey1, inchikey2, tanimoto_score = self.selected_inchikey_pairs[pair_index]
        spectrum1 = self._get_pos_spectrum_with_inchikey(inchikey1, self.random_nr_generator)
        spectrum2 = self._get_neg_spectrum_with_inchikey(inchikey2, self.random_nr_generator)
        self._idx += 1
        return spectrum1, spectrum2, tanimoto_score

    def __len__(self):
        return len(self.selected_inchikey_pairs)

    def __str__(self):
        return f"SpectrumPairGenerator with {len(self.selected_inchikey_pairs)} pairs available"

    def get_scores(self):
        if self._compact_schedule:
            return np.asarray(self.selected_inchikey_pairs.scores).tolist()
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
        matching_spectrum_ids = self.pos_spectra_by_inchikey.get(inchikey)
        if matching_spectrum_ids is None or len(matching_spectrum_ids) == 0:
            raise ValueError(
                "No matching positive-mode inchikey found "
                "(note: expected first 14 characters), likely switched pos and neg in entry"
            )
        return self.spectra_pos[random_number_generator.choice(matching_spectrum_ids)]

    def _get_neg_spectrum_with_inchikey(self, inchikey: str, random_number_generator) -> Spectrum:
        matching_spectrum_ids = self.neg_spectra_by_inchikey.get(inchikey)
        if matching_spectrum_ids is None or len(matching_spectrum_ids) == 0:
            raise ValueError(
                "No matching negative-mode inchikey found "
                "(note: expected first 14 characters), likely switched pos and neg in entry"
            )
        return self.spectra_neg[random_number_generator.choice(matching_spectrum_ids)]


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
