from typing import List

import numpy as np
import pandas as pd
import torch
from matchms import Spectrum
from ms2deepscore.fingerprint_similarity_computations import (
    compute_fingerprint_similarity_matrix,
    is_dense_fingerprint_type,
)
from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator
from ms2deepscore.models import SiameseSpectralModel
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.inchikey_pair_selection import compute_fingerprints_for_training
from ms2deepscore.vector_operations import cosine_similarity_matrix


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

    def __init__(
        self,
        spectrums: List[Spectrum],
        ms2ds_model: SiameseSpectralModel,
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
        self.inchikey14s = [s.get("inchikey")[:14] for s in spectrums]  # type: ignore
        self.ms2ds_model = ms2ds_model
        self.device = device
        self.ms2ds_model.to(self.device)
        self.indexes = np.arange(len(self.spectrums))
        self.batch_size = self.settings.evaluator_distribution_size
        self.fingerprints, fingerprint_inchikeys = compute_fingerprints_for_training(
            self.spectrums,
            fingerprint_type=self.ms2ds_model.model_settings.fingerprint_type,
            nbits=self.ms2ds_model.model_settings.fingerprint_nbits,
        )

        self.fingerprint_index = {
            inchikey: idx
            for idx, inchikey in enumerate(fingerprint_inchikeys)
        }

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
        indexes = self.indexes[batch_index * batch_size : ((batch_index + 1) * batch_size)]

        spec_tensors, meta_tensors = tensorize_spectra(
            [self.spectrums[i] for i in indexes], self.ms2ds_model.model_settings
        )
        with torch.no_grad():
            embeddings = self.ms2ds_model.encoder(
                spec_tensors.to(self.device),
                meta_tensors.to(self.device),
            )

        embeddings_numpy = embeddings.cpu().numpy()

        ms2ds_scores = cosine_similarity_matrix(embeddings_numpy, embeddings_numpy)

        # Compute true scores
        fingerprint_type = self.ms2ds_model.model_settings.fingerprint_type

        inchikeys = [self.inchikey14s[i] for i in indexes]
        fingerprint_indexes = [
            self.fingerprint_index[inchikey]
            for inchikey in inchikeys
        ]

        if is_dense_fingerprint_type(fingerprint_type):
            batch_fingerprints = self.fingerprints[fingerprint_indexes]
        else:
            batch_fingerprints = [
                self.fingerprints[i]
                for i in fingerprint_indexes
            ]

        tanimoto_scores = compute_fingerprint_similarity_matrix(
            batch_fingerprints,
            batch_fingerprints,
            fingerprint_type=fingerprint_type,
        )

        return (
            torch.as_tensor(tanimoto_scores),
            torch.as_tensor(ms2ds_scores),
            embeddings.cpu(),
        )

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.rng.shuffle(self.indexes)

    def __getitem__(self, batch_index: int):
        """Generate one batch of data."""
        return self._compute_embeddings_and_scores(batch_index)

    def compute_fingerprint_dataframe(
        self,
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
            spectrums, fingerprint_type, fingerprint_nbits
        )

        fingerprints_df = pd.DataFrame(fingerprints, index=inchikeys14_unique)
        return fingerprints_df
