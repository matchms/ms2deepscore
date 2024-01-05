import numpy as np
from matplotlib import pyplot as plt


def tanimoto_dependent_losses(predictions, true_values, ref_score_bins):
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).

    Parameters
    ----------
    predictions
        Scores that should be evaluated
    true_values
        Reference scores (= ground truth).
    ref_score_bins
        Bins for the refernce score to evaluate the performance of scores.
    """
    bin_content = []
    rmses = []
    # maes = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    for i in range(len(ref_scores_bins_inclusive) - 1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i + 1]
        bounds.append((low, high))
        idx = np.where((true_values >= low) & (true_values < high))
        bin_content.append(idx[0].shape[0])
        # maes.append(np.abs(true_values[idx] - predictions[idx]).mean())
        rmses.append(np.sqrt(np.square(true_values[idx] - predictions[idx]).mean()))
    return bin_content, bounds, rmses


def plot_rmse_per_bin(predicted_scores, true_scores):
    ref_score_bins = np.linspace(0, 1.0000001, 11)
    bin_content, bounds, rmses = tanimoto_dependent_losses(predicted_scores, true_scores, ref_score_bins)

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
                                          labels):
    """Combines the plot of multiple comparisons into one plot

    """
    ref_score_bins = np.linspace(0, 1.0000001, 11)
    if not len(list_of_true_values) == len(list_of_true_values) == len(labels):
        raise ValueError("The number of predicted scores and true values should be equal.")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   figsize=(8, 6), dpi=120)
    for i, true_values in enumerate(list_of_true_values):
        bin_content, bounds, rmses = tanimoto_dependent_losses(list_of_predicted_scores[i],
                                                               true_values,
                                                               ref_score_bins)
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
    plt.xticks(np.arange(len(ref_score_bins) - 1),
               [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
