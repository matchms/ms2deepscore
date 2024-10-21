from typing import List

import numpy as np
from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import PredictionsAndTanimotoScores


def bin_dependent_losses(average_loss_per_inchikey_pair,
                         true_values,
                         ref_score_bins,
                         ):
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).

    Parameters
    ----------
    average_loss_per_inchikey_pair
        Precalculated average loss per inchikey pair (this can be any loss type)
    true_values
        Reference scores (= ground truth).
    ref_score_bins
        Bins for the reference score to evaluate the performance of scores. in the form [(0.0, 0.1), (0.1, 0.2) ...]

    """
    if average_loss_per_inchikey_pair.shape != true_values.shape:
        raise ValueError("Expected true values and predictions to have the same shape")
    bin_content = []
    losses = []
    bounds = []
    for i, (low, high) in enumerate(ref_score_bins):
        bounds.append((low, high))
        if i == 0:
            idx = np.where((true_values >= low) & (true_values <= high))
        else:
            idx = np.where((true_values > low) & (true_values <= high))
        if idx[0].shape[0] == 0:
            raise ValueError("No reference scores within bin")
        bin_content.append(idx[0].shape[0])
        # Add values
        losses.append(average_loss_per_inchikey_pair.iloc[idx].mean().mean())
    return bin_content, bounds, losses


def plot_loss_per_bin(predictions_and_tanimoto_scores: PredictionsAndTanimotoScores,
                      ref_score_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(0, 10)]),
                      loss_type="RMSE"):

    bin_content, bounds, rmses = bin_dependent_losses(
        average_loss_per_inchikey_pair=predictions_and_tanimoto_scores.get_loss_per_inchikey_pair(loss_type),
        true_values=predictions_and_tanimoto_scores.tanimoto_df,
        ref_score_bins=ref_score_bins,)
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 5), dpi=120)

    ax1.plot(np.arange(len(rmses)), rmses, "o:", color="crimson")
    ax1.set_title(loss_type)
    ax1.set_ylabel(loss_type)
    ax1.grid(True)

    ax2.plot(np.arange(len(rmses)), bin_content, "o:", color="teal")
    ax2.set_title('# of spectrum pairs')
    ax2.set_ylabel("# of spectrum pairs")
    ax2.set_xlabel("Tanimoto score bin")
    ax2.set_ylim(bottom=0)
    plt.xticks(np.arange(len(rmses)),
               [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
    ax2.grid(True)
    plt.tight_layout()


def plot_loss_per_bin_multiple_benchmarks(list_of_predictions_and_tanimoto_scores: List[PredictionsAndTanimotoScores],
                                          nr_of_bins=10,
                                          loss_type="RMSE"):
    """Combines the plot of multiple comparisons into one plot
    """
    ref_score_bins = np.array([(x / nr_of_bins, x / nr_of_bins + 1/nr_of_bins) for x in range(nr_of_bins)])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   figsize=(8, 6), dpi=120)
    labels = []
    for predictions_and_tanimoto_scores in list_of_predictions_and_tanimoto_scores:
        bin_content, bounds, rmses = bin_dependent_losses(
            predictions_and_tanimoto_scores.get_loss_per_inchikey_pair(loss_type),
            predictions_and_tanimoto_scores.tanimoto_df,
            ref_score_bins,
            )
        ax1.plot(np.arange(len(rmses)), rmses, "o:")
        ax2.plot(np.arange(len(rmses)), bin_content, "o:")
        labels.append(predictions_and_tanimoto_scores.label)
    fig.legend(labels, loc="center right")
    ax1.set_title(loss_type)
    ax1.set_ylabel(loss_type)
    ax1.grid(True)

    ax2.set_title('# of spectrum pairs')
    ax2.set_ylabel("# of spectrum pairs")
    ax2.set_xlabel("Tanimoto score bin")
    ax2.set_ylim(bottom=0)
    plt.xticks(np.arange(len(ref_score_bins)),
               [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
