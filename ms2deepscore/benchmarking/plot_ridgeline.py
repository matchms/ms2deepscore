import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_combined_ridgeline_plot(reference_scores,
                                   comparison_scores,
                                   n_bins=10,
                                   min_resolution=20,
                                   max_resolution=100,
                                   ref_score_name="Tanimoto similarity",
                                   compare_score_name="MS2DeepScore"):
    """
    Plot ridgeline-style histograms to compare reference and comparison scores.

    Parameters
    ----------
    reference_scores
        Reference score array.
    comparison_scores
        Comparison score array.
    n_bins
        Number of bins. The default is 5.
    min_resolution
        Minimum histogram resolution. The default is 20.
    max_resolution
        The mainimum histogram resolution. The default is 100.
    ref_score_name
        Label string. The default is "Tanimoto similarity".
    compare_score_name
        Label string. The default is "MS2DeepScore".
    """
    # pylint: disable=too-many-arguments

    histograms, used_bins, _, _ = calculate_histograms(reference_scores, comparison_scores,
                                                       n_bins, min_resolution, max_resolution)

    _, (ax_main, ax_hist) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(12, 9))

    # Plot ridge plot on main axis
    ridgeline_plot(histograms, used_bins,
                    xlabel=compare_score_name, ylabel=ref_score_name, ax=ax_main)
    ax_hist.set_title("Comparison of Tanimoto scores and predictions.")

    # Plot histogram of bin numbers on the second axis
    score_histogram(reference_scores, n_bins, ax=ax_hist, ylabel=ref_score_name)
    ax_hist.set_title("Tanimoto score distribution")

    plt.tight_layout()
    plt.show()


def ridgeline_plot(histograms,
                   y_score_bins,
                   xlabel="MS2DeepScore",
                   ylabel="Tanimoto similarity",
                   ax=None):
    """Create a plot of (partly overlapping) distributions based on score comparison, on specified axis.
    """
    if ax is None:
        ax = plt.gca()

    n_bins = len(y_score_bins)
    colors = ["crimson", "lightblue", "teal"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    shift = 0.7
    alpha = 1.0

    for i in range(0, len(histograms)):
        idx = len(histograms)-i-1
        data = histograms[idx][0]
        data = data/max(data)
        ax.fill_between(histograms[idx][1][:-1], -shift*i, [(-shift*i + x) for x in data], color=cmap1(i/10), alpha=alpha)
        ax.plot(histograms[idx][1][:-1], [(-shift*i + x) for x in data], linewidth=2, color="white")
        ax.plot(histograms[idx][1][:-1], [(-shift*i + x) for x in data], ".-", color="gray", alpha=0.5)

    #ax.set_xticks([])
    y_score_bins = [[a, b] for (a, b) in y_score_bins]
    y_score_bins[0][0] = 0
    y_score_bins[-1][1] = 1
    ax.set_yticks(-shift*np.arange(len(histograms)),
                  [f"{a:.1f} to < {b:.1f}" for (a, b) in y_score_bins[::-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 1])
    ax.set_ylim([-(n_bins - 1)*shift, 2*shift])


def compute_bins(n_bins):
    ref_scores_bins= np.linspace(0, 1, n_bins+1)
    ref_scores_bins_inclusive= np.linspace(0, 1, n_bins+1)
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    return ref_scores_bins, ref_scores_bins_inclusive


def score_histogram(scores, n_bins, ax=None, ylabel="scores"):
    if ax is None:
        ax = plt.gca()
    bins, inclusive_bins = compute_bins(n_bins)
    bin_content, _ = np.histogram(scores.flatten(), bins=inclusive_bins)
    ax.hist(bins[:-1], bins, weights=bin_content, orientation='horizontal', color="lightblue", rwidth=0.75)


    ax.set_yticks(bins[1]/2 + bins[:-1],
                  [f"{bins[i]:.1f} to < {bins[i+1]:.1f}" for i in range(len(bin_content))])
    ax.set_xlabel("Number of Pairs")
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, (1 + 1/n_bins)])
    ax.set_xscale("log")
    for i, bin_value in enumerate(bin_content):
        ax.text(0.05, 1/len(bins) * (i + 0.5), #bins[1]* (i + 0.5),
                f"{bin_value} pairs", transform=ax.transAxes, ha="left", va="center")


def calculate_histograms(reference_scores, comparison_scores, n_bins=10, min_resolution=20, max_resolution=100):
    """Calcualte a series of histograms, one for every bin."""
    # pylint: disable=too-many-locals
    def get_hist_bins(resolution):
        hist_bins = np.linspace(0, 1, resolution)
        hist_bins = np.concatenate((hist_bins, np.array([2.0])))
        return hist_bins

    histograms = []
    used_bins = []
    bin_content = []
    resolutions = []
    ref_scores_bins_inclusive = np.linspace(0, 1, n_bins+1)
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf

    for i in range(n_bins):
        used_bins.append((ref_scores_bins_inclusive[i], ref_scores_bins_inclusive[i+1]))
        idx = np.where((reference_scores >= ref_scores_bins_inclusive[i]) & (reference_scores < ref_scores_bins_inclusive[i+1]))
        bin_content.append(idx[0].shape[0])
        resolution = int(max(min(max_resolution, idx[0].shape[0]/25), min_resolution))
        resolutions.append(resolution)
        hist_bins = get_hist_bins(resolution)
        a, b = np.histogram(comparison_scores[idx], bins=hist_bins)
        histograms.append((a, b))

    return histograms, used_bins, bin_content, resolutions
