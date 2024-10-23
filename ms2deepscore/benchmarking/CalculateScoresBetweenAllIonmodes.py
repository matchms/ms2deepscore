from typing import Tuple, List

import pandas as pd
import numpy as np
import torch
from matchms import Spectrum
from matchms.filtering.metadata_processing.add_fingerprint import _derive_fingerprint_from_smiles
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from tqdm import tqdm

from ms2deepscore import MS2DeepScore
from ms2deepscore.train_new_model.inchikey_pair_selection import select_inchi_for_unique_inchikeys
from ms2deepscore.vector_operations import cosine_similarity_matrix
from ms2deepscore.models.load_model import load_model


class PredictionsAndTanimotoScores:
    def __init__(self, predictions_df, tanimoto_df, symmetric, label=""):
        self.predictions_df = predictions_df
        self.tanimoto_df = tanimoto_df
        self.symmetric = symmetric
        self.label = label
        # remove predicitons between the same spectrum
        if self.symmetric:
            np.fill_diagonal(self.predictions_df.values, np.nan)

        average_prediction_per_inchikey_pair = self._get_average_prediction_per_inchikey_pair()
        self.list_of_average_predictions, self.list_of_tanimoto_scores = self._convert_scores_df_to_list_of_pairs(
            average_prediction_per_inchikey_pair)

    def _get_average_prediction_per_inchikey_pair(self):
        """Takes a matrix with per spectrum predictions and converts it to a df with the average prediction between all inchikeys"""
        # get the mean prediction per inchikey
        df_grouped = self.predictions_df.groupby(self.predictions_df.index).mean()
        df_grouped_columns = df_grouped.groupby(lambda x: x, axis=1).mean()  # Other axis
        return df_grouped_columns

    def _convert_scores_df_to_list_of_pairs(self, average_predictions_per_inchikey: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Takes in two dataframes with inchikeys as index and returns two lists with scores, which correspond to pairs"""
        predictions = []
        tanimoto_scores = []
        for inchikey_1 in average_predictions_per_inchikey.index:
            for inchikey_2 in average_predictions_per_inchikey.columns:
                prediction = average_predictions_per_inchikey[inchikey_2][inchikey_1]
                # don't include pairs where the prediciton is Nan (this is the case when only a pair against itself is available)
                if not np.isnan(prediction):
                    tanimoto = self.tanimoto_df[inchikey_2][inchikey_1]
                    predictions.append(prediction)
                    tanimoto_scores.append(tanimoto)
        return predictions, tanimoto_scores

    def _get_average_over_inchikey_pairs(self, losses: pd.DataFrame):
        grouped_losses = losses.groupby(losses.index).mean()
        average_losses = grouped_losses.groupby(lambda x: x, axis=1).mean()
        return average_losses

    def get_average_loss_per_bin_per_inchikey_pair(self,
                                                   loss_type: str,
                                                   tanimoto_bins):
        """Calculates the loss per tanimoto bin.
        First the average prediction for each inchikey pair is calculated,
        by taking the average over all predictions between spectra matching these inchikeys"""
        loss_type = loss_type.lower()
        if loss_type == "mse" or loss_type == "rmse":
            losses_per_spectrum_pair = self._get_squared_error_per_spectrum_pair()
        elif loss_type == "mae":
            losses_per_spectrum_pair = self._get_absolute_error_per_spectrum_pair()
        elif loss_type == 'risk_mse':
            losses_per_spectrum_pair = self._get_risk_aware_squared_error_per_spectrum_pair()
        elif loss_type == 'risk_mae':
            losses_per_spectrum_pair = self._get_risk_aware_absolute_error_per_spectrum_pair()
        else:
            raise ValueError(f"The given loss type: {loss_type} is not a valid loss type, choose from mse, "
                             f"rmse, mae, risk_mse and risk_mae")
        average_losses_per_inchikey_pair = self._get_average_over_inchikey_pairs(losses_per_spectrum_pair)

        bin_content, bounds, average_loss_per_bin = self.get_average_loss_per_bin(average_losses_per_inchikey_pair,
                                                                                  tanimoto_bins)
        if loss_type == "RMSE":
            average_loss_per_bin = [average_loss ** 0.5 for average_loss in average_loss_per_bin]
        return bin_content, bounds, average_loss_per_bin

    def _get_absolute_error_per_spectrum_pair(self):
        losses = abs(self.predictions_df - self.tanimoto_df)
        return losses

    def _get_squared_error_per_spectrum_pair(self):
        losses = (self.predictions_df - self.tanimoto_df) ** 2
        return losses

    def _get_risk_aware_squared_error_per_spectrum_pair(self):
        """MSE weighted by target position on scale 0 to 1.
        """
        errors = self.tanimoto_df - self.predictions_df
        errors = torch.sign(errors) * errors ** 2
        uppers = self.tanimoto_df * errors
        lowers = (self.tanimoto_df - 1) * errors
        # Get the max in upper or lower
        risk_aware_squared_error = lowers.combine(uppers, func=max)
        return risk_aware_squared_error

    def _get_risk_aware_absolute_error_per_spectrum_pair(self):
        """MAE weighted by target position on scale 0 to 1.
        """
        errors = self.tanimoto_df - self.predictions_df
        uppers = self.tanimoto_df * errors
        lowers = (self.tanimoto_df - 1) * errors
        # get the max in upper or lower
        risk_aware_absolute_error = lowers.combine(uppers, func=max)
        return risk_aware_absolute_error

    def get_average_loss_per_bin(self,
                                 average_loss_per_inchikey_pair: pd.DataFrame,
                                 ref_score_bins: List[Tuple[float, float]]):
        """Compute average loss per tanimoto score bin

        Parameters
        ----------
        average_loss_per_inchikey_pair
            Precalculated average loss per inchikey pair (this can be any loss type)
        ref_score_bins
            Bins for the reference score to evaluate the performance of scores. in the form [(0.0, 0.1), (0.1, 0.2) ...]
        """
        bin_content = []
        losses = []
        bounds = []
        for i, (low, high) in enumerate(sorted(ref_score_bins)):
            bounds.append((low, high))
            if i == 0:
                # The lowest bin should start including that bin
                idx = np.where((self.tanimoto_df >= low) & (self.tanimoto_df <= high))
            else:
                idx = np.where((self.tanimoto_df > low) & (self.tanimoto_df <= high))
            if idx[0].shape[0] == 0:
                raise ValueError("No reference scores within bin")
            bin_content.append(idx[0].shape[0])
            # Add values
            losses.append(average_loss_per_inchikey_pair.iloc[idx].mean().mean())
        return bin_content, bounds, losses


class CalculateScoresBetweenAllIonmodes:
    """Calculates the true tanimoto scores and average ms2deepscore between unique inchikeys """

    def __init__(self,
                 model_file_name, positive_validation_spectra, negative_validation_spectra):
        self.model_file_name = model_file_name
        self.postive_validation_spectra = positive_validation_spectra
        self.negative_validation_spectra = negative_validation_spectra
        self.model = MS2DeepScore(load_model(model_file_name))

        self.pos_vs_neg_scores = self.get_tanimoto_and_prediction_pairs(
            positive_validation_spectra, negative_validation_spectra, label="positive vs negative")
        self.pos_vs_pos_scores = self.get_tanimoto_and_prediction_pairs(
            positive_validation_spectra, label="positive vs positive")
        self.neg_vs_neg_scores = self.get_tanimoto_and_prediction_pairs(
            negative_validation_spectra, label="negative vs negative")
        # Avoid memory leakage
        torch.cuda.empty_cache()
        del self.model

    def get_tanimoto_and_prediction_pairs(self, spectra_1, spectra_2=None, label="") -> PredictionsAndTanimotoScores:
        symmetric = False
        if spectra_2 is None:
            spectra_2 = spectra_1
            symmetric = True
        if symmetric:
            predictions_df = create_embedding_matrix_symmetric(self.model, spectra_1)
        else:
            predictions_df = create_embedding_matrix_not_symmetric(self.model, spectra_1, spectra_2)
        tanimoto_scores_df = calculate_tanimoto_scores_unique_inchikey(spectra_1, spectra_2)
        return PredictionsAndTanimotoScores(predictions_df, tanimoto_scores_df, symmetric, label)

    def list_of_predictions_and_tanimoto_scores(self):
        return [self.pos_vs_pos_scores,
                self.pos_vs_neg_scores,
                self.neg_vs_neg_scores, ]


# Functions for creating predictions and true value matrix
def create_embedding_matrix_symmetric(model, spectra):
    print("Calculating embeddings")
    embeddings = model.get_embedding_array(spectra)
    print("Calculating similarity between embeddings")
    predictions = cosine_similarity_matrix(embeddings, embeddings)
    # Select the inchikeys per spectrum
    inchikeys = []
    for spectrum in spectra:
        inchikeys.append(spectrum.get("inchikey")[:14])
    # create dataframe with inchikeys as indexes
    predictions_df = pd.DataFrame(predictions, index=inchikeys, columns=inchikeys)
    return predictions_df


def create_embedding_matrix_not_symmetric(model, spectra, spectra_2):
    print("Calculating embeddings")
    embeddings1 = model.get_embedding_array(spectra)
    embeddings2 = model.get_embedding_array(spectra_2)
    print("Calculating similarity between embeddings")

    predictions = cosine_similarity_matrix(embeddings1, embeddings2)
    # Select the inchikeys per spectrum
    inchikeys1 = [spectrum.get("inchikey")[:14] for spectrum in spectra]
    inchikeys2 = [spectrum.get("inchikey")[:14] for spectrum in spectra_2]

    # create dataframe with inchikeys as indexes
    predictions_df = pd.DataFrame(predictions, index=inchikeys1, columns=inchikeys2)
    return predictions_df


def calculate_tanimoto_scores_unique_inchikey(list_of_spectra_1: List[Spectrum],
                                              list_of_spectra_2: List[Spectrum],
                                              fingerprint_type="daylight",
                                              nbits=2048
                                              ):
    """Returns a dataframe with the tanimoto scores between each unique inchikey in list of spectra"""

    def get_fingerprint(smiles: str):
        fingerprint = _derive_fingerprint_from_smiles(smiles,
                                                      fingerprint_type=fingerprint_type,
                                                      nbits=nbits)
        assert isinstance(fingerprint, np.ndarray), f"Fingerprint could not be set smiles is {smiles}"
        return fingerprint

    if len(list_of_spectra_1) == 0 or len(list_of_spectra_2) == 0:
        raise ValueError("The nr of spectra to calculate tanimoto scores should be larger than 0")

    spectra_with_most_frequent_inchi_per_inchikey_1, unique_inchikeys_1 = \
        select_inchi_for_unique_inchikeys(list_of_spectra_1)
    spectra_with_most_frequent_inchi_per_inchikey_2, unique_inchikeys_2 = \
        select_inchi_for_unique_inchikeys(list_of_spectra_2)

    list_of_smiles_1 = [spectrum.get("smiles") for spectrum in spectra_with_most_frequent_inchi_per_inchikey_1]
    list_of_smiles_2 = [spectrum.get("smiles") for spectrum in spectra_with_most_frequent_inchi_per_inchikey_2]

    fingerprints_1 = np.array([get_fingerprint(spectrum) for spectrum in tqdm(list_of_smiles_1,
                                                                              desc="Calculating fingerprints")])
    fingerprints_2 = np.array([get_fingerprint(spectrum) for spectrum in tqdm(list_of_smiles_2,
                                                                              desc="Calculating fingerprints")])
    print("Calculating tanimoto scores")
    tanimoto_scores = jaccard_similarity_matrix(fingerprints_1, fingerprints_2)
    tanimoto_df = pd.DataFrame(tanimoto_scores, index=unique_inchikeys_1, columns=unique_inchikeys_2)
    return tanimoto_df