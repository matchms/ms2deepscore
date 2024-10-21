from typing import Tuple, List

import pandas as pd
import numpy as np
import torch

from ms2deepscore import MS2DeepScore
from ms2deepscore.benchmarking.calculate_scores_for_validation import calculate_tanimoto_scores_unique_inchikey
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

    def get_average_MAE_per_inchikey_pair(self):
        loss = abs(self.predictions_df - self.tanimoto_df)
        grouped_losses = loss.groupby(loss.index).mean()
        average_losses = grouped_losses.groupby(lambda x: x, axis=1).mean()
        return average_losses

    def get_average_MSE_per_inchikey_pair(self):
        loss = (self.predictions_df - self.tanimoto_df) ** 2
        grouped_losses = loss.groupby(loss.index).mean()
        average_mse = grouped_losses.groupby(lambda x: x, axis=1).mean()
        return average_mse

    def get_average_RMSE_per_inchikey_pair(self):
        return self.get_average_MSE_per_inchikey_pair() ** 0.5

    def get_loss_per_inchikey_pair(self, loss_type):
        if loss_type not in ("RMSE", "MSE", "MAE"):
            raise ValueError(f'The loss type {loss_type} is not implemented choose from ("RMSE", "MSE", "MAE")')

        if loss_type == "RMSE":
            return self.get_average_RMSE_per_inchikey_pair()
        if loss_type == "MSE":
            return self.get_average_MSE_per_inchikey_pair()
        if loss_type == "MAE":
            return self.get_average_MAE_per_inchikey_pair()


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

    def get_tanimoto_and_prediction_pairs(self, spectra_1, spectra_2=None, label="") -> PredictionsAndTanimotoScores:
        symmetric = False
        if spectra_2 is None:
            spectra_2 = spectra_1
            symmetric = True
        if symmetric:
            predictions_df = self.create_embedding_matrix_symmetric(spectra_1)
        else:
            predictions_df = self.create_embedding_matrix_not_symmetric(spectra_1, spectra_2)
        tanimoto_scores_df = calculate_tanimoto_scores_unique_inchikey(spectra_1, spectra_2)
        return PredictionsAndTanimotoScores(predictions_df, tanimoto_scores_df, symmetric, label)

    # Functions for creating predictions and true value matrix
    def create_embedding_matrix_symmetric(self, spectra):
        print("Calculating embeddings")
        embeddings = self.model.get_embedding_array(spectra)
        print("Calculating similarity between embeddings")
        predictions = cosine_similarity_matrix(embeddings, embeddings)
        # Select the inchikeys per spectrum
        inchikeys = []
        for spectrum in spectra:
            inchikeys.append(spectrum.get("inchikey")[:14])
        # create dataframe with inchikeys as indexes
        predictions_df = pd.DataFrame(predictions, index=inchikeys, columns=inchikeys)
        return predictions_df

    def create_embedding_matrix_not_symmetric(self, spectra, spectra_2):
        print("Calculating embeddings")
        embeddings1 = self.model.get_embedding_array(spectra)
        embeddings2 = self.model.get_embedding_array(spectra_2)
        print("Calculating similarity between embeddings")

        predictions = cosine_similarity_matrix(embeddings1, embeddings2)
        # Select the inchikeys per spectrum
        inchikeys1 = [spectrum.get("inchikey")[:14] for spectrum in spectra]
        inchikeys2 = [spectrum.get("inchikey")[:14] for spectrum in spectra_2]

        # create dataframe with inchikeys as indexes
        predictions_df = pd.DataFrame(predictions, index=inchikeys1, columns=inchikeys2)
        return predictions_df

    def list_of_predictions_and_tanimoto_scores(self):
        return [self.pos_vs_pos_scores,
                self.pos_vs_neg_scores,
                self.neg_vs_neg_scores, ]
