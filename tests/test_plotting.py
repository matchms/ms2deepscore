import numpy as np
from ms2deepscore.plotting import (calculate_histograms,
                                   create_confusion_matrix_plot,
                                   create_histograms_plot, plot_histograms)


mock_reference_scores = np.random.random((100, 100))
mock_comparison_scores = np.random.random((100, 100))


def test_create_confusion_matrix_plot():
    fig = create_confusion_matrix_plot(mock_reference_scores, mock_comparison_scores)
    assert fig is not None
    assert fig.get_axes()[0].get_xlabel() == "MS2DeepScore"
    assert fig.get_axes()[0].get_ylabel() == 'Tanimoto similarity'


def test_create_histograms_plot():
    # Just checking if it runs without errors
    create_histograms_plot(mock_reference_scores, mock_comparison_scores)
    assert True


def test_calculate_histograms():
    histograms, used_bins, bin_content = calculate_histograms(mock_reference_scores, mock_comparison_scores, n_bins=5, hist_resolution=10)
    
    # Ensure the number of histograms, used bins and bin contents are the same
    assert len(histograms) == len(used_bins) == len(bin_content) == 5

    # Ensure the used bins are valid
    for (low, high) in used_bins:
        assert low <= high

    # Ensure histograms are properly formed
    for histogram in histograms:
        assert len(histogram[0]) == len(histogram[1]) - 1  # histogram frequencies and bin edges

    # Ensure all reference scores are accounted for
    assert sum(bin_content) == mock_reference_scores.shape[0] * mock_reference_scores.shape[1]


def test_plot_histograms():
    # Test the plotting function
    histograms, used_bins, bin_content = calculate_histograms(mock_reference_scores, mock_comparison_scores, n_bins=5, hist_resolution=10)
    plot_histograms(histograms, used_bins, bin_content)
    assert True
