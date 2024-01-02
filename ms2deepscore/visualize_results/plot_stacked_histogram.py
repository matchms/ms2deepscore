from typing import Tuple, List

import numpy as np
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
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, nr_of_bins),
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