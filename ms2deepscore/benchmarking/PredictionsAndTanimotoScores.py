from typing import Tuple, List

import numpy as np
import pandas as pd
import torch

from ms2deepscore.SettingsMS2Deepscore import validate_bin_order


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
                                                   tanimoto_bins: np.ndarray):
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
                                 ref_score_bins: np.ndarray):
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
        validate_bin_order(ref_score_bins)
        ref_score_bins.sort()
        for low, high in ref_score_bins:
            bounds.append((low, high))
            idx = np.where((self.tanimoto_df > low) & (self.tanimoto_df <= high))
            if idx[0].shape[0] == 0:
                bin_content.append(0)
                losses.append(0)
                print(f"There are no scores within the bin {low} - {high}")
            else:
                bin_content.append(idx[0].shape[0])
                # Add values
                losses.append(average_loss_per_inchikey_pair.iloc[idx].mean().mean())
        return bin_content, bounds, losses