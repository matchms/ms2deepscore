import numpy as np
from matplotlib import pyplot as plt


def create_confusion_matrix_plot(reference_scores,
                                 comparison_scores,
                                 n_bins=5,
                                 ref_score_name="Tanimoto similarity",
                                 compare_score_name="MS2DeepScore",
                                 max_square_size=5000,
                                 lower_bound=0, upper_bound=1,
                                 color_by_reference_fraction=True):
    """
    Plot histograms to compare reference and comparison scores.

    Parameters
    ----------
    reference_scores
        Reference score array.
    comparison_scores
        Comparison score array.
    n_bins
        Number of bins. The default is 5.
    ref_score_name
        Label string. The default is "Tanimoto similarity".
    compare_score_name
        Label string. The default is "MS2DeepScore".
    max_square_size
        Maximum square size.
    lower_bound
        Lower bound for scores to include in plot (scores < lower_bound will be added to lowest bin).
    upper_bound
        Upper bound for scores to include in plot
    color_by_reference_fraction
        When True, color squares by fractions within the reference score bin. Default is True.
    """
    # pylint: disable=too-many-arguments
    confusion_like_matrix, confusion_like_matrix_scatter = derive_scatter_data(reference_scores,
                                                                               comparison_scores,
                                                                               lower_bound, upper_bound,
                                                                               n_bins, n_bins)
    fig = plot_confusion_like_matrix(confusion_like_matrix_scatter, confusion_like_matrix,
                                     xlabel=compare_score_name, ylabel=ref_score_name,
                                     max_size=max_square_size,
                                     lower_bound=lower_bound, upper_bound=upper_bound,
                                     color_by_reference_fraction=color_by_reference_fraction)
    return fig


def plot_confusion_like_matrix(confusion_like_matrix_scatter,
                               confusion_like_matrix,
                               xlabel,
                               ylabel,
                               max_size=5000,
                               lower_bound=0,
                               upper_bound=1,
                               color_by_reference_fraction=True):
    """Do the actual plotting"""
    # pylint: disable=too-many-arguments
    summed_tanimoto = []
    for i in range(confusion_like_matrix.shape[0]):
        for _ in range(confusion_like_matrix.shape[1]):
            summed_tanimoto.append(confusion_like_matrix[i,:].sum())

    sizes = np.array([x[2] for x in confusion_like_matrix_scatter])
    colors = 100*sizes/np.array(summed_tanimoto)  # color percentage
    sizes = sizes/np.max(sizes)

    # plt.style.use('seaborn-white')
    if color_by_reference_fraction:
        fig = plt.figure(figsize=(10, 8))
        plt.scatter([x[1] for x in confusion_like_matrix_scatter],
                    [x[0] for x in confusion_like_matrix_scatter], marker='s', c=colors, cmap="plasma",
                    s=sizes*max_size)
    else:
        fig = plt.figure(figsize=(8, 8))
        plt.scatter([x[1] for x in confusion_like_matrix_scatter],
                    [x[0] for x in confusion_like_matrix_scatter], marker='s', c="teal",
                    s=sizes*max_size)
    if color_by_reference_fraction:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('% within respective Tanimoto bin', rotation=90, fontsize=14)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(lower_bound, upper_bound)
    plt.ylim(lower_bound, upper_bound)
    plt.clim(0)
    plt.grid(True)
    return fig


def derive_scatter_data(reference_scores,
                        comparison_scores,
                        lower_bound,
                        upper_bound,
                        n_bins_x, n_bins_y):
    """Helper function to collect actual scatter plot data"""
    # pylint: disable=too-many-arguments
    bins_x = np.linspace(lower_bound,upper_bound+0.0001, n_bins_x+1)
    bins_y = np.linspace(lower_bound,upper_bound+0.0001, n_bins_y+1)
    confusion_like_matrix = np.zeros((n_bins_x, n_bins_y))
    confusion_like_matrix_scatter = []
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            if i > 0:
                low_x = bins_x[i]
            else:
                low_x = np.min(reference_scores)
            if j > 0:
                low_y = bins_y[j]
            else:
                low_y = np.min(comparison_scores)
            idx = np.where((reference_scores>=low_x) & (reference_scores<bins_x[i+1]) &
                          (comparison_scores>=low_y) & (comparison_scores<bins_y[j+1]))
            confusion_like_matrix[i, j] = idx[0].shape[0]
            confusion_like_matrix_scatter.append(((bins_x[i] + bins_x[i+1])/2,
                                                 (bins_y[j] + bins_y[j+1])/2,
                                                 idx[0].shape[0]))
    return confusion_like_matrix, confusion_like_matrix_scatter


def tanimoto_dependent_losses(scores, scores_ref, ref_score_bins):
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).

    Parameters
    ----------

    scores
        Scores that should be evaluated
    scores_ref
        Reference scores (= ground truth).
    ref_score_bins
        Bins for the refernce score to evaluate the performance of scores.
    """
    bin_content = []
    rmses = []
    maes = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    for i in range(len(ref_scores_bins_inclusive) - 1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i + 1]
        bounds.append((low, high))
        idx = np.where((scores_ref >= low) & (scores_ref < high))
        bin_content.append(idx[0].shape[0])
        maes.append(np.abs(scores_ref[idx] - scores[idx]).mean())
        rmses.append(np.sqrt(np.square(scores_ref[idx] - scores[idx]).mean()))
    return bin_content, bounds, rmses, maes


def plot_rmse_per_bin(predicted_scores, true_scores):
    ref_score_bins = np.linspace(0, 1.0, 11)
    bin_content, bounds, rmses, _ = tanimoto_dependent_losses(
        predicted_scores,
        true_scores,
        ref_score_bins)

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 5), dpi=120)

    ax1.plot(np.arange(len(rmses)), rmses, "o:", color="crimson")
    ax1.set_title('RMSE')
    ax1.set_ylabel("RMSE")
    ax1.grid(True)

    ax2.plot(np.arange(len(rmses)), bin_content, "o:", color="teal")
    ax2.set_title('# of spectrum pairs')
    ax2.set_ylabel("# of spectrum pairs")
    ax2.set_xlabel("Tanimoto score bin")
    plt.yscale('log')
    plt.xticks(np.arange(len(rmses)),
               [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
    ax2.grid(True)
    plt.tight_layout()
