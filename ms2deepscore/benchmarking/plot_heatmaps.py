from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.validation_loss_calculation.PredictionsAndTanimotoScores import PredictionsAndTanimotoScores


def create_3_heatmaps(pairs: CalculateScoresBetweenAllIonmodes, nr_of_bins):
    minimum_y_axis = 0
    maximum_y_axis = 1
    for predictions_and_tanimoto_score in pairs.list_of_predictions_and_tanimoto_scores():
        average_pred_per_inchikey_pair = predictions_and_tanimoto_score.get_average_prediction_per_inchikey_pair()
        minimum = average_pred_per_inchikey_pair.min().min()
        maximum = average_pred_per_inchikey_pair.max().max()
        if minimum < minimum_y_axis:
            minimum_y_axis = minimum
        if maximum > maximum_y_axis:
            maximum_y_axis = maximum

    x_bins = np.linspace(0, 1, nr_of_bins + 1)
    y_bins = np.linspace(minimum_y_axis, maximum_y_axis + 0.00001, nr_of_bins + 1)

    # Take the average per bin
    pos_pos_normalized_heatmap = create_normalized_heatmap_data(pairs.pos_vs_pos_scores, x_bins, y_bins)
    neg_neg_normalized_heatmap = create_normalized_heatmap_data(pairs.neg_vs_neg_scores, x_bins, y_bins)
    pos_neg_normalized_heatmap = create_normalized_heatmap_data(pairs.pos_vs_neg_scores, x_bins, y_bins)

    maximum_heatmap_intensity = max(pos_pos_normalized_heatmap.max(), neg_neg_normalized_heatmap.max(),
                                    pos_neg_normalized_heatmap.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(neg_neg_normalized_heatmap.T, origin='lower', interpolation='nearest',
                   cmap="inferno", vmax=maximum_heatmap_intensity, extent=[0, 1, minimum_y_axis, maximum_y_axis])
    axes[0].set_title("Negative vs negative")
    axes[1].imshow(pos_pos_normalized_heatmap.T, origin='lower', interpolation='nearest',
                   cmap="inferno", vmax=maximum_heatmap_intensity, extent=[0, 1, minimum_y_axis, maximum_y_axis])
    axes[1].set_title("Positive vs positive")
    im2 = axes[2].imshow(pos_neg_normalized_heatmap.T, origin='lower', interpolation='nearest',
                         cmap="inferno", vmax=maximum_heatmap_intensity, extent=[0, 1, minimum_y_axis, maximum_y_axis])
    axes[2].set_title("Positive vs negative")
    for ax in axes:
        ax.set_xlabel("True chemical similarity")
        ax.set_ylabel("Predicted chemical similarity")
        ax.set_xlim(0, 1)
        ax.set_ylim(minimum_y_axis, maximum_y_axis)

    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Density')  # Label for the colorbar
    return fig


def create_normalized_heatmap_data(prediction_and_tanimoto_scores: PredictionsAndTanimotoScores,
                                   x_bins, y_bins):
    average_prediction = \
        prediction_and_tanimoto_scores.get_average_prediction_per_inchikey_pair()
    list_of_tanimoto_scores, list_of_average_predictions = convert_dataframes_to_lists_with_matching_pairs(
        prediction_and_tanimoto_scores.tanimoto_df,
        average_prediction)
    heatmap = np.histogram2d(list_of_tanimoto_scores,
                             list_of_average_predictions,
                             bins=(x_bins, y_bins))[0]
    normalized_heatmap = heatmap / heatmap.sum(axis=1, keepdims=True)
    return normalized_heatmap


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
                predictions.append(prediction)
                tanimoto_scores.append(tanimoto_df[inchikey_2][inchikey_1])
    return tanimoto_scores, predictions
