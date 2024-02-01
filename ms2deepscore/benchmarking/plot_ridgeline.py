import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_combined_ridgeline_plot(reference_scores, comparison_scores, n_bins=10, hist_resolution=100,
                         ref_score_name="Tanimoto similarity", compare_score_name="MS2DeepScore"):
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
    hist_resolution
        Histogram resolution. The default is 100.
    ref_score_name
        Label string. The default is "Tanimoto similarity".
    compare_score_name
        Label string. The default is "MS2DeepScore".

    """
    histograms, used_bins, bin_content = calculate_histograms(reference_scores, comparison_scores, n_bins, hist_resolution)

    fig, (ax_main, ax_hist) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(14, 10))

    # Plot ridge plot on main axis
    ridgeline_plot(histograms, used_bins, bin_content, hist_resolution,
                    xlabel=compare_score_name, ylabel=ref_score_name, ax=ax_main)
    ax_hist.set_title("Comparison of Tanimoto scores and predictions.")

    # Plot histogram of bin numbers on the second axis
    score_histogram(reference_scores, n_bins, ax=ax_hist, ylabel=ref_score_name)
    ax_hist.set_title("Tanimoto score distribution")

    plt.tight_layout()
    plt.show()


def ridgeline_plot(histograms, y_score_bins, bin_content=None,
                   hist_resolution=100, xlabel="MS2DeepScore", ylabel="Tanimoto similarity",
                   ax=None):
    """Create a plot of (partly overlapping) distributions based on score comparison, on specified axis."""
    if ax is None:
        ax = plt.gca()

    colors = ["crimson", "lightblue", "teal"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    shift = 0.7
    alpha = 1.0

    for i in range(0, len(histograms)):
        data = histograms[len(histograms)-i-1][0]
        data = data/max(data)
        ax.fill_between(histograms[0][1][:hist_resolution], -shift*i, [(-shift*i + x) for x in data], color=cmap1(i/10), alpha=alpha)
        ax.plot(histograms[0][1][:hist_resolution], [(-shift*i + x) for x in data], linewidth=2, color="white")
        ax.plot(histograms[0][1][:hist_resolution], [(-shift*i + x) for x in data], ".-", color="gray", alpha=0.5)
        if bin_content:
            ax.text(0.01, -shift*i+shift/6, f"{bin_content[::-1][i]} pairs")

    ax.set_xticks([])
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
    print(bins, bin_content)
    #ax.stairs(bin_content, bins, orientation='horizontal', color="teal")
    ax.hist(bins[:-1], bins, weights=bin_content, orientation='horizontal', color="teal", rwidth=0.8)
    #ax.bar(
    ax.set_xlabel("Number of Pairs")
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, (1 + 1/n_bins)])
    ax.set_xscale("log")


def calculate_histograms(reference_scores, comparison_scores, n_bins=10, hist_resolution=100):
    """Calcualte a series of histograms, one for every bin."""
    hist_bins = np.linspace(0, 1, hist_resolution)
    hist_bins = np.concatenate((hist_bins, np.array([2.0])))

    histograms = []
    used_bins = []
    bin_content = []
    ref_scores_bins_inclusive = np.linspace(0, 1, n_bins+1)
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    
    for i in range(n_bins):
        used_bins.append((ref_scores_bins_inclusive[i], ref_scores_bins_inclusive[i+1]))
        idx = np.where((reference_scores >= ref_scores_bins_inclusive[i]) & (reference_scores < ref_scores_bins_inclusive[i+1]))
        bin_content.append(idx[0].shape[0])
        a, b = np.histogram(comparison_scores[idx], bins=hist_bins)
        histograms.append((a, b))

    return histograms, used_bins, bin_content
