from typing import Tuple, List

import numpy as np
import pandas as pd

from ms2deepscore.SettingsMS2Deepscore import validate_bin_order


class PredictionsAndTanimotoScores:
    """Stores predictions and tanimoto scores and can calculate losses and averages per inchikey pair"""
    def __init__(self, predictions_df: pd.DataFrame,
                 tanimoto_df: pd.DataFrame,
                 symmetric: bool, label=""):
        """
        Parameters
        ----------
        predictions_df:
            A dataframe with predictions between all spectra. The df has to be labeled with inchikeys. If there are
            multiple spectra per inchikey, the inchikey is repeated in rows and columns.
        tanimoto_df:
            A dataframe with the tanimoto scores between inchikeys. These are unique pairs, so inchikeys don't repeat.
        symmetric:
            If the dataframe is symmetric.
        label:
            A label that can be used by plotting functions. For instance "Positive vs Negative".
        """
        self.predictions_df = predictions_df
        self.tanimoto_df = tanimoto_df
        self.symmetric = symmetric
        self.label = label

        self.check_input_data()
        # remove predicitons between the same spectrum
        if self.symmetric:
            np.fill_diagonal(self.predictions_df.values, np.nan)

    def check_input_data(self):
        """Checks that the prediction df and tanimoto df have the expected format"""
        if not isinstance(self.predictions_df, pd.DataFrame) or not isinstance(self.tanimoto_df, pd.DataFrame):
            raise TypeError("Expected a pandas DF as input")

        if not len(np.unique(self.tanimoto_df.index)) == len(self.tanimoto_df.index):
            raise ValueError("The tanimoto df should have unique indices representing the inchikeys")
        if not len(np.unique(self.tanimoto_df.columns)) == len(self.tanimoto_df.columns):
            raise ValueError("The tanimoto df should have unique column indexes representing the inchikeys")

        if np.all(np.unique(self.predictions_df.index).sort() == self.tanimoto_df.index.sort_values()):
            raise ValueError("All predicition indexes should appear at least once in tanimoto df indexes")
        if np.all(np.unique(self.predictions_df.columns).sort() == self.tanimoto_df.columns.sort_values()):
            raise ValueError("All predicition columns should appear at least once in tanimoto df columns")

        if self.symmetric:
            if not np.all(self.predictions_df.index == self.predictions_df.columns):
                raise ValueError("If the setting is symmetric, indexes and columns are expected to be equal")
            if not np.all(self.tanimoto_df.index == self.tanimoto_df.columns):
                raise ValueError("If the setting is symmetric, indexes and columns are expected to be equal")

    def get_average_prediction_per_inchikey_pair(self) -> pd.DataFrame:
        """Gets the average prediction per unique inchikey pair (instead of per spectrum pair)"""
        return get_average_per_inchikey_pair(self.predictions_df)

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
        average_losses_per_inchikey_pair = get_average_per_inchikey_pair(losses_per_spectrum_pair)

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
        errors = np.sign(errors) * errors ** 2
        uppers = self.tanimoto_df * errors
        lowers = (self.tanimoto_df - 1) * errors
        # Get the max in upper or lower
        risk_aware_squared_error = pd.DataFrame(np.maximum(lowers, uppers), columns=uppers.columns)
        return risk_aware_squared_error

    def _get_risk_aware_absolute_error_per_spectrum_pair(self):
        """MAE weighted by target position on scale 0 to 1.
        """
        errors = self.tanimoto_df - self.predictions_df
        uppers = self.tanimoto_df * errors
        lowers = (self.tanimoto_df - 1) * errors
        # get the max in upper or lower
        risk_aware_absolute_error = pd.DataFrame(np.maximum(lowers, uppers),
                                                 columns=lowers.columns)
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


def convert_dataframes_to_lists_with_matching_pairs(tanimoto_df: pd.DataFrame,
                                           average_predictions_per_inchikey_pair: pd.DataFrame
                                           ) -> Tuple[List[float], List[float]]:
    """Takes in two dataframes with inchikeys as index and returns two lists with scores, which correspond to pairs"""
    predictions = []
    tanimoto_scores = []
    for inchikey_1 in average_predictions_per_inchikey_pair.index:
        for inchikey_2 in average_predictions_per_inchikey_pair.columns:
            prediction = average_predictions_per_inchikey_pair[inchikey_2][inchikey_1]
            # don't include pairs where the prediciton is Nan (this is the case when only a pair against itself is available)
            if not np.isnan(prediction):
                tanimoto = tanimoto_df[inchikey_2][inchikey_1]
                predictions.append(prediction)
                tanimoto_scores.append(tanimoto)
    return predictions, tanimoto_scores


def get_average_per_inchikey_pair(df: pd.DataFrame):
    """Calculates the average in a df of everything with the same index and column

    Used to get the average prediction or loss for an inchikey with multiple available spectra (repeating indexes/columns)
    """
    # Group the same inchikeys per index and get the mean
    indexes_grouped = df.groupby(df.index).mean()
    # Group the same inchikeys per column and get the mean
    average_per_inchikey_pair = indexes_grouped.groupby(lambda x: x, axis=1).mean()
    return average_per_inchikey_pair
