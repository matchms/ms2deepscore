import numpy as np
from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes


def create_3_heatmaps(pairs: CalculateScoresBetweenAllIonmodes, nr_of_bins):
    # Get the minimum and maximum prediction (if larger of smaller than 1
    minimum_y_axis = min(min(pairs.pos_vs_neg_scores.list_of_average_predictions),
                         min(pairs.pos_vs_pos_scores.list_of_average_predictions),
                         min(pairs.neg_vs_neg_scores.list_of_average_predictions), 0)
    maximum_y_axis = max(max(pairs.pos_vs_neg_scores.list_of_average_predictions),
                         max(pairs.pos_vs_pos_scores.list_of_average_predictions),
                         max(pairs.neg_vs_neg_scores.list_of_average_predictions), 1)

    x_bins = np.linspace(0, 1, nr_of_bins + 1)
    y_bins = np.linspace(minimum_y_axis, maximum_y_axis + 0.00001, nr_of_bins + 1)

    pos_pos_heatmap = np.histogram2d(pairs.pos_vs_pos_scores.list_of_tanimoto_scores,
                                     pairs.pos_vs_pos_scores.list_of_average_predictions, bins=(x_bins, y_bins))[0]
    neg_neg_heatmap = np.histogram2d(pairs.neg_vs_neg_scores.list_of_tanimoto_scores,
                                     pairs.neg_vs_neg_scores.list_of_average_predictions, bins=(x_bins, y_bins))[0]
    pos_neg_heatmap = np.histogram2d(pairs.pos_vs_neg_scores.list_of_tanimoto_scores,
                                     pairs.pos_vs_neg_scores.list_of_average_predictions, bins=(x_bins, y_bins))[0]

    # Take the average per bin
    pos_pos_normalized_heatmap = pos_pos_heatmap / pos_pos_heatmap.sum(axis=1, keepdims=True)
    neg_neg_normalized_heatmap = neg_neg_heatmap / neg_neg_heatmap.sum(axis=1, keepdims=True)
    pos_neg_normalized_heatmap = pos_neg_heatmap / pos_neg_heatmap.sum(axis=1, keepdims=True)

    maximum_heatmap_intensity = max(pos_pos_normalized_heatmap.max(), neg_neg_normalized_heatmap.max(), pos_neg_normalized_heatmap.max())

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