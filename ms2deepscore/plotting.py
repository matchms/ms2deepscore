import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_histograms_plot(reference_scores, comparison_scores, n_bins=10, hist_resolution=100,
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


def create_confusion_matrix_plot(reference_scores, comparison_scores, n_bins=5,
                                 ref_score_name="Tanimoto similarity", compare_score_name="MS2DeepScore",
                                 max_square_size=5000):
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

    """
    # pylint: disable=too-many-arguments
    confusion_like_matrix, confusion_like_matrix_scatter = derive_scatter_data(reference_scores,
                                                                               comparison_scores,
                                                                               n_bins, n_bins)
    fig = plot_confusion_like_matrix(confusion_like_matrix_scatter, confusion_like_matrix,
                                     xlabel=compare_score_name, ylabel=ref_score_name,
                                     max_size=max_square_size)
    return fig


def plot_confusion_like_matrix(confusion_like_matrix_scatter, confusion_like_matrix,
                              xlabel="MS2DeepScore", ylabel="Tanimoto score", max_size=5000):
    """Do the actual plotting"""
    summed_tanimoto = []
    for i in range(confusion_like_matrix.shape[0]):
        for _ in range(confusion_like_matrix.shape[1]):
            summed_tanimoto.append(confusion_like_matrix[i,:].sum())

    sizes = np.array([x[2] for x in confusion_like_matrix_scatter])
    colors = 100*sizes/np.array(summed_tanimoto)  # color percentage
    sizes = sizes/np.max(sizes)

    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=(10, 8))
    plt.scatter([x[1] for x in confusion_like_matrix_scatter],
               [x[0] for x in confusion_like_matrix_scatter], marker='s', c=colors, cmap="plasma",
               s=sizes*max_size)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('% within respective Tanimoto bin', rotation=90, fontsize=14)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-0, 1)
    plt.ylim(-0, 1)
    plt.clim(0)
    plt.grid(True)
    return fig


def derive_scatter_data(reference_scores, comparison_scores, n_bins_x, n_bins_y):
    bins_x = np.linspace(0,1.0001, n_bins_x+1)
    bins_y = np.linspace(0,1.0001, n_bins_y+1)
    confusion_like_matrix = np.zeros((n_bins_x, n_bins_y))
    confusion_like_matrix_scatter = []
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            idx = np.where((reference_scores>=bins_x[i]) & (reference_scores<bins_x[i+1]) &
                          (comparison_scores>=bins_y[j]) & (comparison_scores<bins_y[j+1]))
            confusion_like_matrix[i, j] = idx[0].shape[0]
            confusion_like_matrix_scatter.append(((bins_x[i] + bins_x[i+1])/2,
                                                 (bins_y[j] + bins_y[j+1])/2,
                                                 idx[0].shape[0]))
    return confusion_like_matrix, confusion_like_matrix_scatter
