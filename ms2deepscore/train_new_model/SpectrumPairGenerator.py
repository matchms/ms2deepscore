""" Data generators for training/inference with MS2DeepScore model.
"""
from typing import List
import numpy as np
import torch
from matchms import Spectrum
from ms2deepscore.SettingsMS2Deepscore import (SettingsMS2Deepscore)
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.InchikeyPairGenerator import InchikeyPairGenerator
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    select_compound_pairs_wrapper)
from ms2deepscore.utils import split_by_ionmode


class SpectrumPairGenerator:
    """Generates data for training a siamese Pytorch model.

    This class provides a data generator specifically designed for training a Siamese Pytorch model with a curated set
    of compound pairs. It takes a InchikeyPairGenerator and randomly selects, augments and tensorizes spectra for each
    inchikey pair.

    By using pre-selected compound pairs (in the InchikeyPairGenerator), this allows more control over the training
    process. The selection of inchikey pairs does not happen in SpectrumPairGenerator (only spectrum selection), but in
    inchikey_pair_selection.py. In inchikey_pair_selection inchikey pairs are picked to balance selected pairs equally
    over different tanimoto score bins to make sure both pairs of similar and dissimilar compounds are sampled.
    In addition inchikeys are selected to occur equally for each pair.
    """

    def __init__(self, spectrums: List[Spectrum],
                 selected_compound_pairs: InchikeyPairGenerator,
                 settings: SettingsMS2Deepscore):
        """Generates data for training a siamese Pytorch model.

        Parameters
        ----------
        spectrums
            List of matchms Spectrum objects.
        selected_compound_pairs
            SelectedCompoundPairs object which contains selected compounds pairs and the
            respective similarity scores.
        settings
            The available settings can be found in SettignsMS2Deepscore
        """
        self.current_batch_index = 0
        self.spectrums = spectrums

        # Collect all inchikeys
        self.spectrum_inchikeys = np.array([s.get("inchikey")[:14] for s in self.spectrums])

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

        unique_inchikeys = np.unique(self.spectrum_inchikeys)
        if len(unique_inchikeys) < self.model_settings.batch_size:
            raise ValueError("The number of unique inchikeys must be larger than the batch size.")
        self.fixed_set = {}

        self.selected_compound_pairs = selected_compound_pairs
        self.inchikey_pair_generator = self.selected_compound_pairs.generator(self.model_settings.shuffle, self.rng)
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
                inchikey1, inchikey2, score = next(self.inchikey_pair_generator)
            except StopIteration as exc:
                raise RuntimeError("The inchikey pair generator is not expected to end, "
                                   "but should instead generate infinite pairs") from exc

            spectrum1 = self._get_spectrum_with_inchikey(inchikey1)
            spectrum2 = self._get_spectrum_with_inchikey(inchikey2)
            yield spectrum1, spectrum2, score

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
            spectra_1 = self._data_augmentation(spectra_1)
            spectra_2 = self._data_augmentation(spectra_2)
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

    def _get_spectrum_with_inchikey(self, inchikey: str) -> Spectrum:
        """
        Get a random spectrum matching the `inchikey` argument.

        NB: A compound (identified by an
        inchikey) can have multiple measured spectrums in a binned spectrum dataset.
        """
        matching_spectrum_id = np.where(self.spectrum_inchikeys == inchikey)[0]
        if len(matching_spectrum_id) <= 0:
            raise ValueError("No matching inchikey found (note: expected first 14 characters)")
        return self.spectrums[self.rng.choice(matching_spectrum_id)]

    def _data_augmentation(self, spectra_tensors):
        for i in range(spectra_tensors.shape[0]):
            spectra_tensors[i, :] = self._data_augmentation_spectrum(spectra_tensors[i, :])
        return spectra_tensors

    def _data_augmentation_spectrum(self, spectrum_tensor):
        """Data augmentation.

        Parameters
        ----------
        spectrum_tensor
            Spectrum in Pytorch tensor form.
        """
        # Augmentation 1: peak removal (peaks < augment_removal_max)
        if self.model_settings.augment_removal_max or self.model_settings.augment_removal_intensity:
            # TODO: Factor out function with documentation + example?

            indices_select = torch.where((spectrum_tensor > 0)
                                         & (spectrum_tensor < self.model_settings.augment_removal_intensity))[0]
            removal_part = self.rng.random(1) * self.model_settings.augment_removal_max
            indices = self.rng.choice(indices_select, int(np.ceil((1 - removal_part) * len(indices_select))))
            if len(indices) > 0:
                spectrum_tensor[indices] = 0

        # Augmentation 2: Change peak intensities
        if self.model_settings.augment_intensity:
            # TODO: Factor out function with documentation + example?
            spectrum_tensor = spectrum_tensor * (
                        1 - self.model_settings.augment_intensity * 2 * (torch.rand(spectrum_tensor.shape) - 0.5))

        # Augmentation 3: Peak addition
        if self.model_settings.augment_noise_max and self.model_settings.augment_noise_max > 0:
            indices_select = torch.where(spectrum_tensor == 0)[0]
            if len(indices_select) > self.model_settings.augment_noise_max:
                indices_noise = self.rng.choice(indices_select,
                                                self.rng.integers(0, self.model_settings.augment_noise_max),
                                                replace=False,
                                                )
            spectrum_tensor[indices_noise] = self.model_settings.augment_noise_intensity * torch.rand(
                len(indices_noise))
        return spectrum_tensor


def create_data_generator(training_spectra,
                          settings,
                          json_save_file=None) -> SpectrumPairGenerator:
    selected_compound_pairs_training = select_compound_pairs_wrapper(training_spectra, settings=settings)
    inchikey_pair_generator = InchikeyPairGenerator(selected_compound_pairs_training)

    if json_save_file is not None:
        inchikey_pair_generator.save_as_json(json_save_file)
    # Create generators
    train_generator = SpectrumPairGenerator(spectrums=training_spectra,
                                            selected_compound_pairs=inchikey_pair_generator,
                                            settings=settings)
    return train_generator
