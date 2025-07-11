from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from ms2deepscore.utils import validate_bin_order


class PredictionsAndTanimotoScores:
    """Stores predictions and Tanimoto scores and can calculate losses and averages per inchikey pair.
    """
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
        # Set every negative prediction to 0
        self.predictions_df = predictions_df.mask(predictions_df < 0, 0)
        self.tanimoto_df = tanimoto_df
        self.symmetric = symmetric
        self.label = label

        self._check_input_data()
        # remove predictions between the same spectrum
        if self.symmetric:
            np.fill_diagonal(self.predictions_df.values, np.nan)

    def _check_input_data(self):
        """Checks that the prediction_df and tanimoto_df have the expected format.
        """
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
        """Gets the average prediction per unique inchikey pair (instead of per spectrum pair).
        """
        return get_average_per_inchikey_pair(self.predictions_df)

    def get_average_loss_per_bin_per_inchikey_pair(self,
                                                   loss_type: str,
                                                   tanimoto_bins: np.ndarray):
        """Calculates the loss per Tanimoto bin.

        First the average prediction for each inchikey pair is calculated,
        by taking the average of all predictions between spectra matching these inchikeys.
        """
        loss_type = loss_type.lower()
        if loss_type == "mse" or loss_type == "rmse":
            losses_per_spectrum_pair = self._get_squared_error_per_spectrum_pair()
        elif loss_type == "mae":
            losses_per_spectrum_pair = self._get_absolute_error_per_spectrum_pair()
        elif loss_type == "risk_mse":
            losses_per_spectrum_pair = self._get_risk_aware_squared_error_per_spectrum_pair()
        elif loss_type == "risk_mae":
            losses_per_spectrum_pair = self._get_risk_aware_absolute_error_per_spectrum_pair()
        else:
            raise ValueError(f"The given loss type: {loss_type} is not a valid loss type, choose from mse, "
                             f"rmse, mae, risk_mse and risk_mae")
        average_losses_per_inchikey_pair = get_average_per_inchikey_pair(losses_per_spectrum_pair)

        bin_content, average_loss_per_bin = self.get_average_per_bin(average_losses_per_inchikey_pair, tanimoto_bins)
        if loss_type == "rmse":
            average_loss_per_bin = [average_loss ** 0.5 for average_loss in average_loss_per_bin]
        return bin_content, average_loss_per_bin

    def _get_absolute_error_per_spectrum_pair(self):
        """Calculates the absolute error

        Used to get the MAE, but the mean is taken after binning over the tanimoto bins.
        """
        losses = np.abs(self.predictions_df - self.tanimoto_df)
        return losses

    def _get_squared_error_per_spectrum_pair(self):
        """Calculates the squared errors

        Used to get the MSE or RMSE, but the mean is taken after binning over the Tanimoto bins.
        """
        losses = (self.predictions_df - self.tanimoto_df) ** 2
        return losses

    def _get_risk_aware_squared_error_per_spectrum_pair(self):
        """Squared error weighted by target position on scale 0 to 1.
        """
        errors = self.tanimoto_df - self.predictions_df
        errors = np.sign(errors) * errors ** 2
        uppers = self.tanimoto_df * errors
        lowers = (self.tanimoto_df - 1) * errors
        # Get the max in upper or lower
        risk_aware_squared_error = pd.DataFrame(np.maximum(lowers, uppers), columns=uppers.columns)
        return risk_aware_squared_error

    def _get_risk_aware_absolute_error_per_spectrum_pair(self):
        """Absolute errors weighted by target position on scale 0 to 1.
        """
        errors = self.tanimoto_df - self.predictions_df
        uppers = self.tanimoto_df * errors
        lowers = (self.tanimoto_df - 1) * errors
        # get the max in upper or lower
        risk_aware_absolute_error = pd.DataFrame(np.maximum(lowers, uppers),
                                                 columns=lowers.columns)
        return risk_aware_absolute_error

    def get_average_per_bin(self,
                            average_per_inchikey_pair: pd.DataFrame,
                            tanimoto_bins: np.ndarray) -> Tuple[List[float], List[float]]:
        """Compute average loss per Tanimoto score bin

        Parameters
        ----------
        average_per_inchikey_pair
            Precalculated average (prediction or loss) per inchikey pair
        ref_score_bins
            Bins for the reference score to evaluate the performance of scores. in the form [(0.0, 0.1), (0.1, 0.2) ...]
        """
        average_predictions = average_per_inchikey_pair.to_numpy()
        validate_bin_order(tanimoto_bins)

        sorted_bins = sorted(tanimoto_bins, key=lambda b: b[0])

        bins = [bin_pair[0] for bin_pair in sorted_bins]
        bins.append(sorted_bins[-1][1])

        digitized = np.digitize(self.tanimoto_df, bins, right=True)
        average_per_bin = []
        nr_of_pairs_in_bin = []
        for i, bin_edges in tqdm(enumerate(sorted_bins), desc="Selecting available inchikey pairs per bin"):
            row_idxs, col_idxs = np.where(digitized == i+ 1)

            predictions_in_this_bin = average_predictions[row_idxs, col_idxs]
            nr_of_pairs_in_bin.append(len(predictions_in_this_bin))
            if len(predictions_in_this_bin) == 0:
                average_per_bin.append(0)
                print(f"The bin between {bin_edges[0]} - {bin_edges[1]}does not have any pairs")
            else:
                average_per_bin.append(np.nanmean(predictions_in_this_bin))
        return nr_of_pairs_in_bin, average_per_bin


def get_average_per_inchikey_pair(df: pd.DataFrame):
    """Calculates the average in a df of everything with the same index and column

    Used to get the average prediction or loss for an inchikey with multiple available spectra (repeating indexes/columns)
    """
    # Group the same inchikeys per index and get the mean
    indexes_grouped = df.groupby(df.index).mean()

    # Group the same inchikeys per column and get the mean
    average_per_inchikey_pair = indexes_grouped.T.groupby(df.columns).mean().T
    return average_per_inchikey_pair
