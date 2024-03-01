""" Data generators for training/inference with MS2DeepScore model.
"""
from typing import List
import numpy as np
import torch
from matchms import Spectrum
from matchms.similarity.vector_similarity_functions import \
    jaccard_similarity_matrix
from ms2deepscore.SettingsMS2Deepscore import (SettingsEmbeddingEvaluator,
                                               SettingsMS2Deepscore)
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.spectrum_pair_selection import (
    SelectedCompoundPairs, compute_fingerprint_dataframe)
from ms2deepscore.vector_operations import cosine_similarity_matrix


class DataGeneratorPytorch:
    """Generates data for training a siamese Pytorch model.

    This class provides a data generator specifically
    designed for training a Siamese Pytorch model with a curated set of compound pairs.
    It uses pre-selected compound pairs, allowing more control over the training process,
    particularly in scenarios where certain compound pairs are of specific interest or
    have higher significance in the training dataset.
    """

    def __init__(self, spectrums: List[Spectrum],
                 selected_compound_pairs: SelectedCompoundPairs,
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
        self.current_index = 0
        self.spectrums = spectrums

        # Collect all inchikeys
        self.spectrum_inchikeys = np.array([s.get("inchikey")[:14] for s in self.spectrums])

        # Set all other settings to input (or otherwise to defaults):
        self.model_settings = settings

        # Initialize random number generator
        if self.model_settings.use_fixed_set:
            if selected_compound_pairs.shuffling:
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
        self.on_epoch_end()

    def __len__(self):
        return int(self.model_settings.num_turns) \
               * int(np.ceil(len(self.selected_compound_pairs.scores) / self.model_settings.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < self.__len__():
            batch = self.__getitem__(self.current_index)
            self.current_index += 1
            return batch
        self.current_index = 0  # make generator executable again
        self.on_epoch_end()
        raise StopIteration

    def _spectrum_pair_generator(self, batch_index: int):
        """Use the provided SelectedCompoundPairs object to pick pairs."""
        batch_size = self.model_settings.batch_size
        indexes = self.indexes[batch_index * batch_size:(batch_index + 1) * batch_size]
        for index in indexes:
            inchikey1 = self.selected_compound_pairs.idx_to_inchikey[index]
            score, inchikey2 = self.selected_compound_pairs.next_pair_for_inchikey(inchikey1)
            spectrum1 = self._get_spectrum_with_inchikey(inchikey1)
            spectrum2 = self._get_spectrum_with_inchikey(inchikey2)
            yield (spectrum1, spectrum2, score)

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.tile(np.arange(len(self.selected_compound_pairs.scores)), int(self.model_settings.num_turns))
        if self.model_settings.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        if self.model_settings.use_fixed_set and batch_index in self.fixed_set:
            return self.fixed_set[batch_index]
        if self.model_settings.random_seed is not None and batch_index == 0:
            self.rng = np.random.default_rng(self.model_settings.random_seed)
        spectrum_pairs = self._spectrum_pair_generator(batch_index)
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


class DataGeneratorEmbeddingEvaluation:
    """Generates data for training an embedding evaluation model.

    This class provides a data for the training of an embedding evaluation model.
    It follows a simple strategy: iterate through all spectra and randomly pick another
    spectrum for comparison. This will not compensate the usually drastic biases
    in Tanimoto similarity and is hence not meant for training the prediction of those
    scores.
    The purpose is rather to show a high number of spectra to a model to learn
    embedding evaluations.

    Spectra are sampled in groups of size batch_size. Before every epoch the indexes are
    shuffled at random. For selected spectra the tanimoto scores, ms2deepscore scores and
    embeddings are returned.
    """

    def __init__(self, spectrums: List[Spectrum],
                 ms2ds_model,
                 settings: SettingsEmbeddingEvaluator,
                 device="cpu",
                 ):
        """

        Parameters
        ----------
        spectrums
            List of matchms Spectrum objects.
        settings
            The available settings can be found in SettignsMS2Deepscore
        """
        self.current_index = 0
        self.settings = settings
        self.spectrums = spectrums
        self.inchikey14s = [s.get("inchikey")[:14] for s in spectrums]
        self.ms2ds_model = ms2ds_model
        self.device = device
        self.ms2ds_model.to(self.device)
        self.indexes = np.arange(len(self.spectrums))
        self.batch_size = self.settings.evaluator_distribution_size
        self.fingerprint_df = compute_fingerprint_dataframe(self.spectrums,
                                                            fingerprint_type=self.ms2ds_model.model_settings.fingerprint_type,
                                                            fingerprint_nbits=self.ms2ds_model.model_settings.fingerprint_nbits)

        # Initialize random number generator
        self.rng = np.random.default_rng(self.settings.random_seed)

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.spectrums) / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < self.__len__():
            batch = self.__getitem__(self.current_index)
            self.current_index += 1
            return batch
        self.current_index = 0  # make generator executable again
        self.on_epoch_end()
        raise StopIteration

    def _compute_embeddings_and_scores(self, batch_index: int):
        batch_size = self.batch_size
        indexes = self.indexes[batch_index * batch_size:((batch_index + 1) * batch_size)]

        spec_tensors, meta_tensors = tensorize_spectra([self.spectrums[i] for i in indexes],
                                                       self.ms2ds_model.model_settings)
        embeddings = self.ms2ds_model.encoder(spec_tensors.to(self.device), meta_tensors.to(self.device))

        ms2ds_scores = cosine_similarity_matrix(embeddings.cpu().detach().numpy(), embeddings.cpu().detach().numpy())

        # Compute true scores
        inchikeys = [self.inchikey14s[i] for i in indexes]
        fingerprints = self.fingerprint_df.loc[inchikeys].to_numpy()

        tanimoto_scores = jaccard_similarity_matrix(fingerprints, fingerprints)

        return torch.tensor(tanimoto_scores), torch.tensor(ms2ds_scores), embeddings.cpu().detach()

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.rng.shuffle(self.indexes)

    def __getitem__(self, batch_index: int):
        """Generate one batch of data.
        """
        return self._compute_embeddings_and_scores(batch_index)
