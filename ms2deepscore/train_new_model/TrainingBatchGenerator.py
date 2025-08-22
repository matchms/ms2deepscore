""" Data generators for training/inference with MS2DeepScore model.
"""
import numpy as np
import torch
from ms2deepscore.SettingsMS2Deepscore import (SettingsMS2Deepscore)
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.InchikeyPairGenerator import InchikeyPairGenerator
from ms2deepscore.train_new_model.data_augmentation import data_augmentation
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    select_compound_pairs_wrapper)


class TrainingBatchGenerator:
    """Generates data for training a siamese Pytorch model.

    This class provides a data generator specifically designed for training a Siamese Pytorch model with a curated set
    of compound pairs. It takes a InchikeyPairGenerator and randomly selects, augments and tensorizes spectra for each
    inchikey pair.

    By using pre-selected compound pairs (in the InchikeyPairGenerator), this allows more control over the training
    process. The selection of inchikey pairs does not happen in TrainingBatchGenerator (only spectrum selection), but in
    inchikey_pair_selection.py. In inchikey_pair_selection inchikey pairs are picked to balance selected pairs equally
    over different tanimoto score bins to make sure both pairs of similar and dissimilar compounds are sampled.
    In addition inchikeys are selected to occur equally for each pair.
    """

    def __init__(self,
                 selected_compound_pairs: InchikeyPairGenerator,
                 settings: SettingsMS2Deepscore):
        """Generates data for training a siamese Pytorch model.

        Parameters
        ----------
        selected_compound_pairs
            SelectedCompoundPairs object which contains selected compounds pairs and the
            respective similarity scores.
        settings
            The available settings can be found in SettignsMS2Deepscore
        """
        self.current_batch_index = 0

        # Set all other settings to input (or otherwise to defaults):
        self.model_settings = settings

        # Initialize random number generator
        if self.model_settings.use_fixed_set:
            if self.model_settings.shuffle:
                raise ValueError(
                    "The generator cannot run reproducibly when shuffling is on for `SelectedCompoundPairs`.")
            if self.model_settings.random_seed is None:
                self.model_settings.random_seed = 0
        self.rng = np.random.default_rng(self.model_settings.random_seed)
        self.inchikey_pair_generator = selected_compound_pairs.generator(self.model_settings.shuffle, self.rng)
        unique_inchikeys = np.unique(selected_compound_pairs.spectrum_inchikeys)
        if len(unique_inchikeys) < self.model_settings.batch_size:
            raise ValueError("The number of unique inchikeys must be larger than the batch size.")
        self.fixed_set = {}

        self.nr_of_batches = int(self.model_settings.num_turns) * int(np.ceil(len(unique_inchikeys) /
                                                                              self.model_settings.batch_size))

    def __len__(self):
        return self.nr_of_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch_index < self.nr_of_batches:
            batch = self.__getitem__(self.current_batch_index)
            self.current_batch_index += 1
            return batch
        self.current_batch_index = 0  # make generator executable again
        raise StopIteration

    def _spectrum_pair_generator(self):
        """Use the provided SelectedCompoundPairs object to pick pairs."""
        for _ in range(self.model_settings.batch_size):
            try:
                spectrum1, spectrum2, score = next(self.inchikey_pair_generator)
                yield spectrum1, spectrum2, score
            except StopIteration as exc:
                raise RuntimeError("The inchikey pair generator is not expected to end, "
                                   "but should instead generate infinite pairs") from exc


    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        if self.model_settings.use_fixed_set and batch_index in self.fixed_set:
            return self.fixed_set[batch_index]
        if self.model_settings.random_seed is not None and batch_index == 0:
            self.rng = np.random.default_rng(self.model_settings.random_seed)
        spectrum_pairs = self._spectrum_pair_generator()
        spectra_1, spectra_2, meta_1, meta_2, targets = self._tensorize_all(spectrum_pairs)

        if self.model_settings.use_fixed_set:
            # Store batches for later epochs
            self.fixed_set[batch_index] = (spectra_1, spectra_2, meta_1, meta_2, targets)
        else:
            spectra_1 = data_augmentation(spectra_1, self.model_settings, self.rng)
            spectra_2 = data_augmentation(spectra_2, self.model_settings, self.rng)
        return spectra_1, spectra_2, meta_1, meta_2, targets

    def _tensorize_all(self, spectrum_pairs):
        spectra_1 = []
        spectra_2 = []
        targets = []
        for pair in spectrum_pairs:
            spectra_1.append(pair[0])
            spectra_2.append(pair[1])
            targets.append(pair[2])

        binned_spectra_1, metadata_1 = tensorize_spectra(spectra_1, self.model_settings)
        binned_spectra_2, metadata_2 = tensorize_spectra(spectra_2, self.model_settings)
        return binned_spectra_1, binned_spectra_2, metadata_1, metadata_2, torch.tensor(targets, dtype=torch.float32)


def create_data_generator(training_spectra,
                          settings,
                          json_save_file=None) -> TrainingBatchGenerator:
    # todo actually create, both between and across ionmodes.
    # pos_spectra, neg_spectra = split_by_ionmode(training_spectra)

    selected_compound_pairs_training = select_compound_pairs_wrapper(training_spectra, settings=settings)
    inchikey_pair_generator = InchikeyPairGenerator(selected_compound_pairs_training, training_spectra)

    if json_save_file is not None:
        inchikey_pair_generator.save_as_json(json_save_file)
    # todo possibly create a single TrainingBatchGenerator which takes in 3 generators and pos and neg spectra to iteratively select each one.
    # Create generators
    # todo also make sure that the TrainingBatchGenerator can work across ionmodes.
    train_generator = TrainingBatchGenerator(selected_compound_pairs=inchikey_pair_generator,
                                             settings=settings)
    return train_generator
