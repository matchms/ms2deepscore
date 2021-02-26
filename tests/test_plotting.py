import numpy as np
from ms2deepscore.plotting import create_confusion_matrix_plot


def test_create_confusion_matrix_plot():
    reference_scores = np.random.random((100, 100))
    comparison_scores = np.random.random((100, 100))
    fig = create_confusion_matrix_plot(reference_scores, comparison_scores)
    assert fig is not None
    assert fig.get_axes()[0].get_xlabel() == "MS2DeepScore"
    assert fig.get_axes()[0].get_ylabel() == 'Tanimoto similarity'
