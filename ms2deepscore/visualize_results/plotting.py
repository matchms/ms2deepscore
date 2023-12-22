import numpy as np
from typing import Tuple, List
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_stacked_histogram_plot_wrapper(tanimoto_scores: np.array,
                                        ms2deepscore_predictions: np.array,
                                        n_bins,
                                        ms2deepscore_nr_of_bin_correction=1.0):
    """Create histogram based score comparison.

        Parameters
    ----------
    tanimoto_scores
        A numpy array matrix with the tanimoto scores
    ms2deepscore_predictions
        A numpy array matrix with the ms2deepscore predictions
    n_bins
        Number of bins. The default is 5.
    ms2deepscore_nr_of_bin_correction
        The number of bins used for the ms2deepscore bins is automated based on the number of pairs in a tanimoto bin,
        to make it always fit in the figure. By changing ms2deepscore_nr_of_bin_correction, the starting nr of bins for
        the ms2deepscore histograms is changed.
    """
    if tanimoto_scores.shape != ms2deepscore_predictions.shape:
        raise ValueError("Expected the predictions and the true values to have the same shape")
    if tanimoto_scores.max() > 1 or tanimoto_scores.min() < 0:
        raise ValueError("The tanimoto score predictions are not between 0 and 1. "
                         "Ms2deepscore predictions and tanimoto score predictions might be accidentally reversed")
    tanimoto_bins = np.linspace(0, 1, n_bins + 1)
    tanimoto_bins[-1] = 1.0000000001

    normalized_counts_per_bin, used_ms2deepscore_bins_per_bin, percentage_of_total_pairs_per_bin = \
        calculate_all_histograms(tanimoto_scores,
                                 ms2deepscore_predictions,
                                 tanimoto_bins,
                                 ms2deepscore_nr_of_bin_correction)

    tanimoto_bin_borders = [(tanimoto_bins[i], tanimoto_bins[i+1])for i in range(len(tanimoto_bins)-1)]
    plot_stacked_histogram(normalized_counts_per_bin, used_ms2deepscore_bins_per_bin,
                           percentage_of_total_pairs_per_bin, tanimoto_bin_borders)


def plot_stacked_histogram(normalized_counts_per_bin,
                           used_ms2deepscore_bins_per_bin,
                           percentage_of_total_pairs_per_bin,
                           tanimoto_bin_borders):
    nr_of_bins = len(normalized_counts_per_bin)
    if len(used_ms2deepscore_bins_per_bin) != nr_of_bins or len(percentage_of_total_pairs_per_bin) != nr_of_bins:
        raise ValueError("The nr of tanimoto bins for each of the input values should be equal")

    # Setup plotting stuff
    color_map = LinearSegmentedColormap.from_list("mycmap", ["teal", "lightblue", "crimson"])

    # Create plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, nr_of_bins),
                             gridspec_kw={'width_ratios': [4, 1]})
    # Loop over each bin.
    for bin_idx in range(0, nr_of_bins):
        normalized_counts = normalized_counts_per_bin[bin_idx]
        used_bin_borders = used_ms2deepscore_bins_per_bin[bin_idx]
        percentage_of_pairs_in_this_bin = percentage_of_total_pairs_per_bin[bin_idx]

        # Fill_between fills the area between two lines.
        # We use this to create a histogram, this approach made it possible to stack multiple histograms.
        # The histogram position is the height of the plot in the entire figure.
        histogram_position = bin_idx
        axes[0].fill_between(
            x=used_bin_borders,
            y1=histogram_position,  # This specifies the bottom line, the histogram is plotted between y1 and y2
            y2=[(histogram_position + y) for y in np.append(normalized_counts, 0)],
            # We add a 0 add the end of the normalized counts, so it is equal in len to used_bin_border,
            # since we use step="post", this combined will result in plotting the histogram correctly
            color=color_map(bin_idx / nr_of_bins),
            alpha=1.0,
            step="post"  # Using step results in a look like a histogram.
        )
        # Print labels with percentage of pairs in each bin.
        axes[0].text(x=0,
                     y=bin_idx + 0.2,  # The height of the text. We add 0.2 to the bin_idx (determined height of hist).
                     s=f"{percentage_of_pairs_in_this_bin:.2f} %")

    # This will add an invisible line on top, making sure the alignment of the stacked histograms is correct.
    axes[0].fill_between([0, 1], nr_of_bins, nr_of_bins, color="white")
    # Add bin sizes as labels
    axes[0].set_yticks(np.arange(nr_of_bins), [f"{a:.2f} to < {b:.2f}" for (a, b) in tanimoto_bin_borders], fontsize=14)
    axes[0].set_xlabel("MS2Deepscore", fontsize=14)
    axes[0].set_ylabel("Tanimoto similarity", fontsize=14)
    axes[0].tick_params(axis="x", labelsize=14)

    # Create a bargraph that shows the percentages per bin
    axes[1].barh(np.arange(nr_of_bins), percentage_of_total_pairs_per_bin,
                 tick_label="", height=0.9,)
    axes[1].set_xlabel("% of pairs", fontsize=14)
    plt.tight_layout()


def calculate_all_histograms(tanimoto_scores,
                             ms2deepscore_predictions,
                             tanimoto_bins,
                             ms2deepscore_nr_of_bin_correction=1.0
                             ) -> Tuple[List[np.array], List[np.array], List[float]]:
    """Calcualte a series of histograms, one for every bin."""
    total_nr_of_pairs = (tanimoto_scores.shape[0] * tanimoto_scores.shape[1])

    normalized_counts_per_bin = []
    used_ms2deepscore_bins_per_bin = []
    percentage_of_total_pairs_per_bin = []
    for i in range(len(tanimoto_bins) - 1):
        indexes_within_bin = np.where((tanimoto_scores >= tanimoto_bins[i]) & (tanimoto_scores < tanimoto_bins[i + 1]))

        # Adjust the hist_resolution based on the nr_of_pairs in the bin
        nr_of_pairs = indexes_within_bin[0].shape[0]
        nr_of_ms2deepscore_bins = int((nr_of_pairs/ms2deepscore_nr_of_bin_correction)**0.5)

        normalized_counts, used_ms2deepscore_bins, total_count = calculate_histogram(ms2deepscore_predictions,
                                                                                     nr_of_ms2deepscore_bins,
                                                                                     indexes_within_bin)
        normalized_counts_per_bin.append(normalized_counts)
        used_ms2deepscore_bins_per_bin.append(used_ms2deepscore_bins)
        percentage_of_total_pairs_per_bin.append(total_count/total_nr_of_pairs * 100)
    return normalized_counts_per_bin, used_ms2deepscore_bins_per_bin, percentage_of_total_pairs_per_bin


def calculate_histogram(ms2deepscore_predictions,
                        nr_of_ms2deepscore_bins,
                        indexes_of_selected_pairs):
    max_ms2deepscore_predictions = ms2deepscore_predictions.max()
    min_ms2deepscore_predictions = ms2deepscore_predictions.min()

    ms2deepscore_bins = np.linspace(min_ms2deepscore_predictions,
                                    max_ms2deepscore_predictions,
                                    nr_of_ms2deepscore_bins + 1)
    counts, used_bins = np.histogram(ms2deepscore_predictions[indexes_of_selected_pairs], bins=ms2deepscore_bins)
    # Normalize the data to have the same area under the curve
    normalized_counts = counts / sum(counts) * nr_of_ms2deepscore_bins
    average_peak_height_after_normalization = 0.08
    normalized_counts = normalized_counts * average_peak_height_after_normalization
    # Prevents the histograms from overlapping each other If it is higher than 1 they would overlap.
    #  It is solved by using larger bins in the histograms.
    if max(normalized_counts) > 0.95:
        nr_of_ms2deepscore_bins = int(nr_of_ms2deepscore_bins / 1.1)
        print(f"One peak was too high, trying {nr_of_ms2deepscore_bins} bins")
        normalized_counts, used_bins, total_count = calculate_histogram(ms2deepscore_predictions,
                                                                        nr_of_ms2deepscore_bins,
                                                                        indexes_of_selected_pairs)
    total_count = sum(counts)
    return normalized_counts, used_bins, total_count


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


if __name__ == "__main__":
    np.random.seed(123)
    dimension = (1500, 1500)
    plot_stacked_histogram_plot_wrapper(np.random.random(dimension) ** 2, np.random.normal(0.5, 10.0, dimension),
                                        n_bins=10)
    plt.show()
