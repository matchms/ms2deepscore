import numpy as np
import pandas as pd

from ms2deepscore import MS2DeepScore
from ms2deepscore.validation_loss_calculation.calculate_scores_for_validation import (
    calculate_tanimoto_scores_unique_inchikey,
)
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.vector_operations import cosine_similarity_matrix
from ms2deepscore.utils import validate_bin_order


class ValidationLossCalculator:
    """A class to calculate validation loss for MS2DeepScore models based on validation spectra and Tanimoto scores.
    """
    def __init__(
        self,
        val_spectrums,
        settings: SettingsMS2Deepscore,
        chunk_size: int = 10_000,
    ):
        """
        Parameters:
        -----------
        val_spectrums:
            A list of spectra used for validation.
        settings:
            Configuration settings for MS2DeepScore.
        chunk_size:
            Number of spectra per chunk used during chunked validation similarity computation.
        """
        self.val_spectrums = val_spectrums
        self.score_bins = settings.same_prob_bins
        self.settings = settings
        self.chunk_size = chunk_size

        self.inchikeys14 = [spectrum.get("inchikey")[:14] for spectrum in self.val_spectrums]
        self.unique_inchikeys14 = sorted(set(self.inchikeys14))

        self.tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
            list_of_spectra_1=self.val_spectrums,
            list_of_spectra_2=None,
            fingerprint_type=self.settings.fingerprint_type,
            nbits=self.settings.fingerprint_nbits
        )

    @staticmethod
    def _chunk_indices(n_items: int, chunk_size: int):
        for start in range(0, n_items, chunk_size):
            stop = min(start + chunk_size, n_items)
            yield start, stop

    def _compute_all_embeddings(self, ms2deepscore_model: MS2DeepScore):
        print("Calculating embeddings for all validation spectra")
        return ms2deepscore_model.get_embedding_array(self.val_spectrums)

    def _compute_prediction_block(self, embeddings, start_i, stop_i, start_j, stop_j) -> pd.DataFrame:
        embeddings_i = embeddings[start_i:stop_i]
        embeddings_j = embeddings[start_j:stop_j]
        predictions = cosine_similarity_matrix(embeddings_i, embeddings_j)

        inchikeys_i = self.inchikeys14[start_i:stop_i]
        inchikeys_j = self.inchikeys14[start_j:stop_j]

        return pd.DataFrame(predictions, index=inchikeys_i, columns=inchikeys_j)

    def _initialize_pair_accumulators(self, loss_types):
        loss_sum_per_type = {}
        loss_count_per_type = {}

        for loss_type in loss_types:
            loss_sum_per_type[loss_type] = pd.DataFrame(
                np.zeros((len(self.unique_inchikeys14), len(self.unique_inchikeys14)), dtype=np.float64),
                index=self.unique_inchikeys14,
                columns=self.unique_inchikeys14,
            )
            loss_count_per_type[loss_type] = pd.DataFrame(
                np.zeros((len(self.unique_inchikeys14), len(self.unique_inchikeys14)), dtype=np.int64),
                index=self.unique_inchikeys14,
                columns=self.unique_inchikeys14,
            )
        return loss_sum_per_type, loss_count_per_type

    @staticmethod
    def _get_block_loss_df(predictions_df: pd.DataFrame, tanimoto_df_block: pd.DataFrame, loss_type: str) -> pd.DataFrame:
        loss_type = loss_type.lower()

        if loss_type in ("mse", "rmse"):
            return (predictions_df - tanimoto_df_block) ** 2

        if loss_type == "mae":
            return np.abs(predictions_df - tanimoto_df_block)

        if loss_type == "risk_mse":
            errors = tanimoto_df_block - predictions_df
            errors = np.sign(errors) * errors ** 2
            uppers = tanimoto_df_block * errors
            lowers = (tanimoto_df_block - 1) * errors
            return lowers.where(lowers >= uppers, uppers)

        if loss_type == "risk_mae":
            errors = tanimoto_df_block - predictions_df
            uppers = tanimoto_df_block * errors
            lowers = (tanimoto_df_block - 1) * errors
            return lowers.where(lowers >= uppers, uppers)

        raise ValueError(
            f"The given loss type: {loss_type} is not a valid loss type, choose from "
            f"mse, rmse, mae, risk_mse and risk_mae"
        )

    @staticmethod
    def _aggregate_block_to_inchikey_pairs(loss_df: pd.DataFrame):
        grouped_sum = loss_df.groupby(loss_df.index).sum()
        grouped_sum = grouped_sum.T.groupby(loss_df.columns).sum().T

        grouped_count = loss_df.groupby(loss_df.index).count()
        grouped_count = grouped_count.T.groupby(loss_df.columns).sum().T

        return grouped_sum, grouped_count

    def _update_global_pair_accumulators(
        self,
        global_sums: pd.DataFrame,
        global_counts: pd.DataFrame,
        block_sum: pd.DataFrame,
        block_count: pd.DataFrame,
    ):
        global_sums.loc[block_sum.index, block_sum.columns] += block_sum
        global_counts.loc[block_count.index, block_count.columns] += block_count

    def _final_average_per_inchikey_pair(
        self,
        loss_sum_df: pd.DataFrame,
        loss_count_df: pd.DataFrame,
    ) -> pd.DataFrame:
        average_df = loss_sum_df.copy()
        average_df[:] = np.nan

        valid = loss_count_df > 0
        average_df[valid] = loss_sum_df[valid] / loss_count_df[valid]
        return average_df

    def _average_per_bin_from_inchikey_pair_df(
        self,
        average_per_inchikey_pair: pd.DataFrame,
        tanimoto_bins: np.ndarray,
        apply_sqrt: bool = False,
    ):
        validate_bin_order(tanimoto_bins)

        average_predictions = average_per_inchikey_pair.to_numpy()
        sorted_bins = sorted(tanimoto_bins, key=lambda b: b[0])
        bins = [bin_pair[0] for bin_pair in sorted_bins]
        bins.append(sorted_bins[-1][1])

        digitized = np.digitize(self.tanimoto_scores, bins, right=True)

        nr_of_pairs_in_bin = []
        average_per_bin = []

        for i, bin_edges in enumerate(sorted_bins):
            row_idxs, col_idxs = np.where(digitized == i + 1)
            predictions_in_this_bin = average_predictions[row_idxs, col_idxs]

            valid_predictions = predictions_in_this_bin[~np.isnan(predictions_in_this_bin)]
            nr_of_pairs_in_bin.append(len(valid_predictions))

            if len(valid_predictions) == 0:
                average_per_bin.append(0.0)
                print(f"The bin between {bin_edges[0]} - {bin_edges[1]} does not have any pairs")
            else:
                avg = np.mean(valid_predictions)
                if apply_sqrt:
                    avg = avg ** 0.5
                average_per_bin.append(avg)

        return nr_of_pairs_in_bin, average_per_bin

    def _compute_global_average_losses_per_inchikey_pair(self, embeddings, loss_types):
        loss_sum_per_type, loss_count_per_type = self._initialize_pair_accumulators(loss_types)
        chunk_ranges = list(self._chunk_indices(len(self.val_spectrums), self.chunk_size))

        for chunk_idx_i, (start_i, stop_i) in enumerate(chunk_ranges):
            for chunk_idx_j in range(chunk_idx_i, len(chunk_ranges)):
                start_j, stop_j = chunk_ranges[chunk_idx_j]

                predictions_df = self._compute_prediction_block(
                    embeddings,
                    start_i,
                    stop_i,
                    start_j,
                    stop_j,
                )

                unique_i = pd.Index(predictions_df.index.unique())
                unique_j = pd.Index(predictions_df.columns.unique())
                tanimoto_df_block = self.tanimoto_scores.loc[unique_i, unique_j]

                for loss_type in loss_types:
                    base_loss_type = "mse" if loss_type.lower() == "rmse" else loss_type
                    loss_df = self._get_block_loss_df(predictions_df, tanimoto_df_block, base_loss_type)

                    if chunk_idx_i == chunk_idx_j:
                        np.fill_diagonal(loss_df.values, np.nan)

                    block_sum, block_count = self._aggregate_block_to_inchikey_pairs(loss_df)

                    self._update_global_pair_accumulators(
                        loss_sum_per_type[loss_type],
                        loss_count_per_type[loss_type],
                        block_sum,
                        block_count,
                    )

        average_loss_per_type = {}
        for loss_type in loss_types:
            average_loss_per_type[loss_type] = self._final_average_per_inchikey_pair(
                loss_sum_per_type[loss_type],
                loss_count_per_type[loss_type],
            )

        return average_loss_per_type

    def compute_binned_validation_loss(
        self,
        model: SiameseSpectralModel,
        loss_types,
    ):
        """
        Compute the validation loss for a model based on binned Tanimoto scores.

        Parameters:
        -----------
        model : SiameseSpectralModel
            The Siamese spectral model to be benchmarked.
        loss_types : list
            A list of loss types to calculate (e.g., 'mse', 'mae').
        """
        ms2deepscore_model = MS2DeepScore(model)
        embeddings = self._compute_all_embeddings(ms2deepscore_model)

        average_loss_per_inchikey_pair_per_type = self._compute_global_average_losses_per_inchikey_pair(
            embeddings,
            loss_types,
        )

        losses_per_bin = {}
        for loss_type in loss_types:
            _, average_loss_per_bin = self._average_per_bin_from_inchikey_pair_df(
                average_loss_per_inchikey_pair_per_type[loss_type],
                self.settings.same_prob_bins,
                apply_sqrt=(loss_type.lower() == "rmse"),
            )
            losses_per_bin[loss_type] = average_loss_per_bin

        average_losses = {
            loss_type: np.mean(losses_per_bin[loss_type])
            for loss_type in loss_types
        }

        return average_losses, losses_per_bin
