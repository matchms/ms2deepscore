from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_reversed_stacked_histogram_plot(tanimoto_scores: np.array,
                                         ms2deepscore_predictions: np.array,
                                         title="",
                                         ms2deepscore_nr_of_bin_correction=1.0):
    if tanimoto_scores.max() > 1 or tanimoto_scores.min() < 0:
        raise ValueError("The tanimoto score predictions are not between 0 and 1. "
                         "Ms2deepscore predictions and tanimoto score predictions might be accidentally reversed")
    ms2deepscore_bins = np.array([0, 0.7, 0.85, 1])

    normalized_counts_per_bin, used_ms2deepscore_bins_per_bin, percentage_of_total_pairs_per_bin = \
        calculate_all_histograms(ms2deepscore_predictions, tanimoto_scores, ms2deepscore_bins,
                                 ms2deepscore_nr_of_bin_correction)

    plot_stacked_histogram(normalized_counts_per_bin, used_ms2deepscore_bins_per_bin,
                           percentage_of_total_pairs_per_bin, ms2deepscore_bins,
                           x_label="Tanimoto similarity", y_label="MS2Deepscore", title=title)


def plot_stacked_histogram_plot_wrapper(tanimoto_scores: np.array,
                                        ms2deepscore_predictions: np.array,
                                        n_bins,
                                        title="",
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

    if tanimoto_scores.max() > 1 or tanimoto_scores.min() < 0:
        raise ValueError("The tanimoto score predictions are not between 0 and 1. "
                         "Ms2deepscore predictions and tanimoto score predictions might be accidentally reversed")
    tanimoto_bins = np.linspace(0, 1, n_bins + 1)
    tanimoto_bins[-1] = 1.0000000001

    normalized_counts_per_bin, used_ms2deepscore_bins_per_bin, percentage_of_total_pairs_per_bin = \
        calculate_all_histograms(tanimoto_scores, ms2deepscore_predictions, tanimoto_bins,
                                 ms2deepscore_nr_of_bin_correction)

    plot_stacked_histogram(normalized_counts_per_bin, used_ms2deepscore_bins_per_bin, percentage_of_total_pairs_per_bin,
                           tanimoto_bins, "MS2Deepscore", "Tanimoto similarity", title)


def plot_stacked_histogram(normalized_counts_per_bin,
                           used_x_axis_bins_per_bin,
                           percentage_of_total_pairs_per_bin,
                           bins_y_axis,
                           x_label,
                           y_label,
                           title):
    """Creates a stacked histogram"""
    # pylint: disable=too-many-arguments
    nr_of_bins = len(normalized_counts_per_bin)
    if len(used_x_axis_bins_per_bin) != nr_of_bins or len(percentage_of_total_pairs_per_bin) != nr_of_bins:
        raise ValueError("The nr of tanimoto bins for each of the input values should be equal")

    # Setup plotting stuff
    color_map = LinearSegmentedColormap.from_list("mycmap", ["teal", "lightblue", "crimson"])

    # Create plot
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, nr_of_bins),
                           gridspec_kw={'width_ratios': [4, 1]})

    # Create a bargraph that shows the percentages per bin
    axes[1].barh(np.arange(nr_of_bins), percentage_of_total_pairs_per_bin,
                 tick_label="", height=0.9, )
    axes[1].set_xlabel("% of pairs", fontsize=14)

    # Plot the stacked histograms
    for bin_idx in range(0, nr_of_bins):
        normalized_counts = normalized_counts_per_bin[bin_idx]
        used_bin_borders = used_x_axis_bins_per_bin[bin_idx]
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
    axes[0].set_yticks(np.arange(nr_of_bins), [f"{bins_y_axis[i]:.2f} to < {bins_y_axis[i+1]:.2f}" for i in range(len(bins_y_axis)-1)], fontsize=14)
    axes[0].tick_params(axis="x", labelsize=14)
    axes[0].set_xlabel(x_label, fontsize=14)
    axes[0].set_ylabel(y_label, fontsize=14)
    plt.suptitle(title)
    plt.tight_layout()


def calculate_all_histograms(scores_y_axis: np.array,
                             scores_x_axis: np.array,
                             bins_splitting_y_axis,
                             x_axis_nr_of_bin_correction=1.0
                             ) -> Tuple[List[np.array], List[np.array], List[float]]:
    """Calcualte a series of histograms, one for every bin.

    scores_y_axis:
        These are seperated into different bins on the y axis. For these a histogram is created.
    scores_x_axis:
        Values for the histograms plotted.
    bins_splitting_y_axis:
        The bins to split the scores_y_axis on the y axis.
    ms2deepscore_nr_of_bin_correction:
        The nr of bins for the x axis is based on the nr of values. This is done to make the histogram look a bit smooth.
        If you have a low number of pairs in a bin, but very precise bins, the result is very noisy and hard to read.
        The ms2deepscore_nr_of_bin_correction, changes the starting point for the number of bins.
    """
    if scores_y_axis.shape != scores_x_axis.shape:
        raise ValueError("Expected the predictions and the true values to have the same shape")
    total_nr_of_pairs = (scores_y_axis.shape[0] * scores_y_axis.shape[1])

    normalized_counts_per_bin = []
    used_ms2deepscore_bins_per_bin = []
    percentage_of_total_pairs_per_bin = []
    for i in range(len(bins_splitting_y_axis) - 1):
        indexes_within_bin = np.where(
            (scores_y_axis >= bins_splitting_y_axis[i]) & (scores_y_axis < bins_splitting_y_axis[i + 1]))

        # Adjust the hist_resolution based on the nr_of_pairs in the bin
        nr_of_pairs = indexes_within_bin[0].shape[0]
        nr_of_ms2deepscore_bins = int((nr_of_pairs / x_axis_nr_of_bin_correction) ** 0.5)

        normalized_counts, used_ms2deepscore_bins, total_count = \
            calculate_histogram_with_max_height(scores_x_axis[indexes_within_bin], nr_of_ms2deepscore_bins,
                                                minimum_bin_value=scores_x_axis.min(),
                                                maximum_bin_value=scores_x_axis.max())
        normalized_counts_per_bin.append(normalized_counts)
        used_ms2deepscore_bins_per_bin.append(used_ms2deepscore_bins)
        percentage_of_total_pairs_per_bin.append(total_count / total_nr_of_pairs * 100)
    return normalized_counts_per_bin, used_ms2deepscore_bins_per_bin, percentage_of_total_pairs_per_bin


def calculate_histogram_with_max_height(input_values: np.array,
                                        starting_nr_of_bins: int,
                                        minimum_bin_value,
                                        maximum_bin_value,
                                        maximum_height=0.95,
                                        average_peak_height_after_normalization=0.08
                                        ):  # pylint: disable=too-many-arguments
    """Creates a histogram, that has a maximum hight of all bins of 0.95

    The reason for the maximum is that this plots nicely as a stacked histogram.

    input_values:
        A numpy array with the values to create a histogram of
    starting_nr_of_bins:
        The number of bins to start at.
        The number of bins will be decreased until the maximum height of the peaks is below 0.95.
    minimal_value:
        The lowest value of the bins. This has to be the same for all the stacked plots.
    maximal_value:
        The highest value of the bins. This has to be the same for all stacked plots.
    maximum_height:
        The maximum height allowed for each peak. The number of bins will be decreased until this is fulfilled.
        By decreasing the number of bins the bins will become wider and thereby the height lower.
    average_peak_height_after_normalization:
        The average peak height of all the peaks.
    """
    counts, used_bins = np.histogram(input_values,
                                     bins=np.linspace(minimum_bin_value,
                                                      maximum_bin_value,
                                                      starting_nr_of_bins + 1))
    # Normalize the data to have the same area under the curve
    normalized_counts = counts / sum(counts) * starting_nr_of_bins
    normalized_counts = normalized_counts * average_peak_height_after_normalization
    # Prevents the histograms from overlapping each other If it is higher than 1 they would overlap.
    #  It is solved by using larger bins in the histograms.
    if len(normalized_counts) > 0:
        if max(normalized_counts) > maximum_height:
            starting_nr_of_bins = int(starting_nr_of_bins / 1.1)
            print(f"One peak was too high, trying {starting_nr_of_bins} bins")
            normalized_counts, used_bins, total_count = \
                calculate_histogram_with_max_height(
                    input_values, starting_nr_of_bins, minimum_bin_value=minimum_bin_value,
                    maximum_bin_value=maximum_bin_value,
                    maximum_height=maximum_height,
                    average_peak_height_after_normalization=average_peak_height_after_normalization)
    total_count = sum(counts)
    return normalized_counts, used_bins, total_count
