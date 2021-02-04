import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_histograms_plot(reference_scores, comparison_scores, n_bins=10, hist_resolution=100,
                          ref_score_name="Tanimoto similarity", compare_score_name="MS2DeepScore"):
    """
    Plot histograms to compare reference and comparison scores.

    Parameters
    ----------
    reference_scores : TYPE
        DESCRIPTION.
    comparison_scores : TYPE
        DESCRIPTION.
    n_bins : TYPE, optional
        DESCRIPTION. The default is 10.
    hist_resolution : TYPE, optional
        DESCRIPTION. The default is 100.
    ref_score_name : TYPE, optional
        DESCRIPTION. The default is "Tanimoto similarity".
    compare_score_name : TYPE, optional
        DESCRIPTION. The default is "MS2DeepScore".

    """
    # pylint: disable=too-many-arguments
    histograms, used_bins, bin_content = calculate_histograms(reference_scores, comparison_scores,
                                                 n_bins, hist_resolution)

    plot_histograms(histograms, used_bins, bin_content, xlabel=compare_score_name, ylabel=ref_score_name)


def plot_histograms(histograms, y_score_bins, bin_content=None, xlabel="MS2DeepScore", ylabel="Tanimoto similarity"):
    """Create histogram based score comparison."""

    # Setup plotting stuff
    colors = ["crimson", "lightblue", "teal"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    plt.style.use('seaborn-white')
    shift = 0.7
    alpha = 1.0 #0.5

    # Create plot
    plt.figure(figsize=(10,10))

    for i in range(0, len(histograms)):
        data = histograms[len(histograms)-i-1][0]
        data = data/max(data)
        plt.fill_between(histograms[0][1][:100], -shift*i, [(-shift*i + x) for x in data], color=cmap1(i/10), alpha=alpha)
        if i > 0:
            plt.plot(histograms[0][1][:100], [(-shift*i + x) for x in data], color="white")
        if bin_content:
            plt.text(0.01, -shift*i+shift/6, f"{bin_content[::-1][i]} pairs")#, color="white")

    plt.yticks(-shift*np.arange(len(histograms)),
               [f"{a:.2f} to < {b:.2f}" for (a, b) in y_score_bins[::-1]], fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim([0, 1])


def calculate_histograms(reference_scores, comparison_scores, n_bins=10, hist_resolution=100):
    """Calcualte a series of histograms, one for every bin."""
    d_bin = 1/n_bins
    hist_bins = np.linspace(0, 1, hist_resolution)
    hist_bins = np.concatenate((hist_bins, np.array([2.0])))

    histograms = []
    used_bins = []
    bin_content = []
    for i in range(n_bins+1):
        used_bins.append((i*d_bin, (i + 1) * d_bin))
        idx = np.where((reference_scores >= i*d_bin) & (reference_scores < (i + 1) * d_bin))
        bin_content.append(idx[0].shape[0])
        a, b = np.histogram(comparison_scores[idx], bins=hist_bins)
        histograms.append((a, b))

    return histograms, used_bins, bin_content
