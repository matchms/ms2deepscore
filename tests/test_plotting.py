import numpy as np
from ms2deepscore.visualize_results.plotting import (
    calculate_all_histograms, create_confusion_matrix_plot, plot_stacked_histogram_plot_wrapper)


mock_reference_scores = np.random.random((100, 100))
mock_comparison_scores = np.random.random((100, 100))


def test_create_confusion_matrix_plot():
    fig = create_confusion_matrix_plot(mock_reference_scores, mock_comparison_scores)
    assert fig is not None
    assert fig.get_axes()[0].get_xlabel() == "MS2DeepScore"
    assert fig.get_axes()[0].get_ylabel() == 'Tanimoto similarity'


def test_calculate_histograms():
    nr_of_bins = 5
    tanimoto_bins = np.linspace(0, 1, nr_of_bins + 1)
    tanimoto_bins[-1] = 1.0000000001
    normalized_counts_per_bin, used_ms2deepscore_bins_per_bin, percentage_of_total_pairs_per_bin = \
        calculate_all_histograms(mock_reference_scores, mock_comparison_scores, tanimoto_bins)
    
    # Ensure the number of histograms, used bins and bin contents are the same
    assert len(normalized_counts_per_bin) == len(used_ms2deepscore_bins_per_bin) == \
           len(percentage_of_total_pairs_per_bin) == nr_of_bins

    # Ensure the used bins are valid
    for ms2deepscore_bins in used_ms2deepscore_bins_per_bin:
        assert np.all(sorted(ms2deepscore_bins) == ms2deepscore_bins)
        assert round(ms2deepscore_bins[0]) == 0
        assert round(ms2deepscore_bins[-1]) == 1
    # Ensure histograms are properly formed
    for i in range(len(normalized_counts_per_bin)):
        assert len(normalized_counts_per_bin[i]) == len(used_ms2deepscore_bins_per_bin[i]) - 1  # histogram frequencies and bin edges


def test_plot_histograms():
    # Test the plotting function
    plot_stacked_histogram_plot_wrapper(mock_reference_scores, mock_comparison_scores, n_bins=5)
    assert True
