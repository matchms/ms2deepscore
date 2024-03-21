import numpy as np
from matplotlib import pyplot as plt
from ms2deepscore.models.loss_functions import bin_dependent_losses


def plot_rmse_per_bin(predicted_scores, true_scores,
                      ref_score_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(0, 10)])):
    bin_content, bounds, losses = bin_dependent_losses(
        predictions=predicted_scores,
        true_values=true_scores,
        ref_score_bins=ref_score_bins,
        loss_types=["rmse"]
        )
    rmses = losses["rmse"]
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 5), dpi=120)

    ax1.plot(np.arange(len(rmses)), rmses, "o:", color="crimson")
    ax1.set_title('RMSE')
    ax1.set_ylabel("RMSE")
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


def plot_rmse_per_bin_multiple_benchmarks(list_of_predicted_scores,
                                          list_of_true_values,
                                          labels,
                                          ref_score_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(0, 10)])):
    """Combines the plot of multiple comparisons into one plot

    """
    if not len(list_of_true_values) == len(list_of_true_values) == len(labels):
        raise ValueError("The number of predicted scores and true values should be equal.")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   figsize=(8, 6), dpi=120)
    for i, true_values in enumerate(list_of_true_values):
        bin_content, bounds, losses = bin_dependent_losses(
            list_of_predicted_scores[i],
            true_values,
            ref_score_bins,
            loss_types=["rmse"]
            )
        rmses = losses["rmse"]
        ax1.plot(np.arange(len(rmses)), rmses, "o:")
        ax2.plot(np.arange(len(rmses)), bin_content, "o:")
    fig.legend(labels, loc="center right")
    ax1.set_title('RMSE')
    ax1.set_ylabel("RMSE")
    ax1.grid(True)

    ax2.set_title('# of spectrum pairs')
    ax2.set_ylabel("# of spectrum pairs")
    ax2.set_xlabel("Tanimoto score bin")
    ax2.set_ylim(bottom=0)
    plt.xticks(np.arange(len(ref_score_bins)),
               [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
