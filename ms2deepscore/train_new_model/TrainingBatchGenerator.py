""" Data generators for training/inference with MS2DeepScore model.
"""
import numpy as np
import torch
from ms2deepscore.SettingsMS2Deepscore import (SettingsMS2Deepscore)
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.SpectrumPairGenerator import SpectrumPairGenerator
from ms2deepscore.train_new_model.data_augmentation import data_augmentation


class TrainingBatchGenerator:
    """Generates data for training a siamese Pytorch model.

    This class provides a data generator specifically designed for training a Siamese Pytorch model with a curated set
    of compound pairs. It takes a SpectrumPairGenerator and augments and tensorizes spectra and combines them into
    batches.

    By using pre-selected compound pairs (in the SpectrumPairGenerator), this allows more control over the training
    process. The selection of inchikey pairs does not happen in SpectrumPairGenerator, but in
    inchikey_pair_selection.py. In inchikey_pair_selection.py inchikey pairs are picked to balance selected pairs equally
    over different tanimoto score bins to make sure both pairs of similar and dissimilar compounds are sampled.
    In addition inchikeys are selected to occur equally for each pair.
    """

    def __init__(self,
                 spectrum_pair_generator: SpectrumPairGenerator,
                 settings: SettingsMS2Deepscore):
        """Generates data for training a siamese Pytorch model.

        Parameters
        ----------
        spectrum_pair_generator
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
        self.spectrum_pair_generator = spectrum_pair_generator
        unique_inchikeys = np.unique(spectrum_pair_generator.spectrum_inchikeys)
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
                spectrum1, spectrum2, score = next(self.spectrum_pair_generator)
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
