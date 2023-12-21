import numpy as np
from typing import Tuple, List
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_histograms(tanimoto_scores,
                    ms2deepscore_predictions,
                    n_bins,
                    normalize_per_bin=True):
    """Create histogram based score comparison.

        Parameters
    ----------
    tanimoto_scores
        The tanimoto scores representing the true values
    ms2deepscore_predictions
        The predicted MS2Deepscore scores
    n_bins
        Number of bins. The default is 5.
    hist_resolution
        Histogram resolution. The default is 100.
    normalize_per_bin
        If true each bin will be normalized to have a similar area under the curve otherwise all bins are normalized in the same way and it reflects
        the frequencies.
    """
    if tanimoto_scores.shape != ms2deepscore_predictions.shape:
        raise ValueError("Expected the predictions and the true values to have the same shape")
    if tanimoto_scores.max() > 1 or tanimoto_scores.min() < 0:
        raise ValueError("The tanimoto score predictions are not between 0 and 1. "
                         "Ms2deepscore predictions and tanimoto score predictions might be accidentally reversed")

    bins = np.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.0000000001

    histogram_per_bin = calculate_histograms(tanimoto_scores,
                                             ms2deepscore_predictions,
                                             bins,
                                             False,
                                             normalize_per_bin)

    # Setup plotting stuff
    color_map = LinearSegmentedColormap.from_list("mycmap", ["teal", "lightblue", "crimson"])
    plot_shifts = np.arange(len(histogram_per_bin))
    alpha = 1.0

    # Create plot
    plt.figure(figsize=(10, len(bins)))
    # Loop over each bin.
    for bin_idx in reversed(range(0, len(histogram_per_bin))):
        normalized_counts, used_bin_borders, total_counts_in_bin = histogram_per_bin[bin_idx]

        # Add the count for the last bin twice
        normalized_counts = np.concatenate((normalized_counts, np.array([0])), axis=0)
        y_levels = [(plot_shifts[bin_idx] + y) for y in normalized_counts]
        plt.fill_between(used_bin_borders,
                         plot_shifts[bin_idx],
                         y_levels,
                         color=color_map(bin_idx / n_bins),
                         alpha=alpha,
                         step="post")

        # Writes down the number of pairs per bin
        percentage_of_all_pairs = total_counts_in_bin/(tanimoto_scores.shape[0]*tanimoto_scores.shape[1]) * 100
        plt.text(ms2deepscore_predictions.min(),
                 plot_shifts[bin_idx] + 0.2, f"{percentage_of_all_pairs:.2f} %")

    plt.xticks(fontsize=14)

    bin_pairs = [(bins[i], bins[i+1])for i in range(len(bins)-1)]
    plt.yticks(plot_shifts,
               [f"{a:.2f} to < {b:.2f}" for (a, b) in bin_pairs], fontsize=14)
    plt.xlabel("MS2Deepscore", fontsize=14)
    plt.ylabel("Tanimoto similarity", fontsize=14)


def calculate_histograms(tanimoto_scores,
                         ms2deepscore_predictions,
                         tanimoto_bins,
                         fixed_nr_of_ms2deepscore_bins=False,
                         normalize_per_bin=True) -> List[Tuple[np.array, np.array, int]]:
    """Calcualte a series of histograms, one for every bin."""
    max_ms2deepscore_predictions = ms2deepscore_predictions.max()
    min_ms2deepscore_predictions = ms2deepscore_predictions.min()
    histogram_per_bin = []

    for i in range(len(tanimoto_bins) - 1):
        indexes_within_bin = np.where((tanimoto_scores >= tanimoto_bins[i]) & (tanimoto_scores < tanimoto_bins[i + 1]))
        # Adjust the hist_resolution based on the nr_of_pairs in the bin
        if fixed_nr_of_ms2deepscore_bins is False:
            nr_of_pairs = indexes_within_bin[0].shape[0]
            nr_of_ms2deepscore_bins = int(nr_of_pairs**0.5)
        else:
            nr_of_ms2deepscore_bins = fixed_nr_of_ms2deepscore_bins
        ms2deepscore_bins = np.linspace(min_ms2deepscore_predictions,
                                        max_ms2deepscore_predictions,
                                        nr_of_ms2deepscore_bins+1)
        counts, used_bins = np.histogram(ms2deepscore_predictions[indexes_within_bin], bins=ms2deepscore_bins)
        # Normalize the data to have the same area under the curve
        if normalize_per_bin:
            normalized_counts = counts / sum(counts) * len(counts) / 8
        else:
            normalized_counts = counts / len(ms2deepscore_predictions) * len(counts) / 2000
        total_count = sum(counts)
        histogram_per_bin.append((normalized_counts, used_bins, total_count))
    return histogram_per_bin


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
