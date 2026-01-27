from typing import List

import numpy as np
from matplotlib import pyplot as plt

from ms2deepscore.validation_loss_calculation.PredictionsAndTanimotoScores import PredictionsAndTanimotoScores
from ms2deepscore.utils import create_evenly_spaced_bins


def plot_loss_per_bin(predictions_and_tanimoto_scores: PredictionsAndTanimotoScores,
                      nr_of_bins=10,
                      loss_type="MSE"):
    ref_score_bins = create_evenly_spaced_bins(nr_of_bins)
    bin_content, rmses = predictions_and_tanimoto_scores.get_average_loss_per_bin_per_inchikey_pair(
        loss_type, ref_score_bins)
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
               [f"{a:.1f} to < {b:.1f}" for (a, b) in ref_score_bins], fontsize=9, rotation='vertical')
    ax2.grid(True)
    plt.tight_layout()


def plot_loss_per_bin_multiple_benchmarks(list_of_predictions_and_tanimoto_scores: List[PredictionsAndTanimotoScores],
                                          nr_of_bins=10,
                                          loss_type="MSE"):
    """Combines the plot of multiple comparisons into one plot
    """
    ref_score_bins = create_evenly_spaced_bins(nr_of_bins)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   figsize=(8, 6), dpi=120)
    labels = []
    for predictions_and_tanimoto_scores in list_of_predictions_and_tanimoto_scores:
        bin_content, rmses = predictions_and_tanimoto_scores.get_average_loss_per_bin_per_inchikey_pair(
            loss_type, ref_score_bins)
        ax1.plot(np.arange(len(rmses)), rmses, "o:")
        ax2.plot(np.arange(len(rmses)), bin_content, "o:")
        labels.append(predictions_and_tanimoto_scores.label)
    fig.legend(labels, loc="center right")
    ax1.set_title(loss_type)
    ax1.set_ylabel(loss_type)
    ax1.grid(True)

    ax2.set_title('# of compound pairs')
    ax2.set_ylabel("# of compound pairs")
    ax2.set_xlabel("Tanimoto score bin")
    ax2.set_ylim(bottom=0)
    plt.xticks(np.arange(len(ref_score_bins)),
               [f"{a:.1f} to < {b:.1f}" for (a, b) in ref_score_bins], fontsize=9, rotation='vertical')
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
