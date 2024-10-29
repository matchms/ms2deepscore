""" Data generators for training/inference with MS2DeepScore model.
"""
import json
from collections import Counter
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from matchms import Spectrum
from matchms.similarity.vector_similarity_functions import \
    jaccard_similarity_matrix
from ms2deepscore.SettingsMS2Deepscore import (SettingsEmbeddingEvaluator,
                                               SettingsMS2Deepscore)
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    select_compound_pairs_wrapper, compute_fingerprints_for_training)
from ms2deepscore.vector_operations import cosine_similarity_matrix


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
                 selected_compound_pairs: "InchikeyPairGenerator",
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
        self.fingerprint_df = self.compute_fingerprint_dataframe(
            self.spectrums,
            fingerprint_type=self.ms2ds_model.model_settings.fingerprint_type,
            fingerprint_nbits=self.ms2ds_model.model_settings.fingerprint_nbits
            )

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

    def compute_fingerprint_dataframe(self,
                                      spectrums: List[Spectrum],
                                      fingerprint_type,
                                      fingerprint_nbits,
                                      ) -> pd.DataFrame:
        """Returns a dataframe with a fingerprints dataframe

        spectrums:
            A list of spectra
        settings:
            The settings that should be used for selecting the compound pairs wrapper. The settings should be specified as a
            SettingsMS2Deepscore object.
        """
        fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
            spectrums,
            fingerprint_type,
            fingerprint_nbits
            )

        fingerprints_df = pd.DataFrame(fingerprints, index=inchikeys14_unique)
        return fingerprints_df


class InchikeyPairGenerator:
    def __init__(self, selected_inchikey_pairs: List[Tuple[str, str, float]]):
        """
        Parameters
        ----------
        selected_inchikey_pairs:
            A list with tuples encoding inchikey pairs like: (inchikey1, inchikey2, tanimoto_score)
        """
        self.selected_inchikey_pairs = selected_inchikey_pairs

    def generator(self, shuffle: bool, random_nr_generator):
        """Infinite generator to loop through all inchikeys.
        After looping through all inchikeys the order is shuffled.
        """
        while True:
            if shuffle:
                random_nr_generator.shuffle(self.selected_inchikey_pairs)

            for inchikey1, inchikey2, tanimoto_score in self.selected_inchikey_pairs:
                yield inchikey1, inchikey2, tanimoto_score

    def __len__(self):
        return len(self.selected_inchikey_pairs)

    def __str__(self):
        return f"InchikeyPairGenerator with {len(self.selected_inchikey_pairs)} pairs available"

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
