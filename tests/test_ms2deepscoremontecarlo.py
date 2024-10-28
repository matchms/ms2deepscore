from pathlib import Path
import numpy as np
import pytest
from ms2deepscore import MS2DeepScoreMonteCarlo
from ms2deepscore.models import load_model
from tests.create_test_spectra import pesticides_test_spectra


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


@pytest.fixture
def ms2_deep_score_instance():
    n_ensembles = 5
    average_type = "median"
    """Load data and models for MS2DeepScore unit tests."""
    spectrums = pesticides_test_spectra()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.pt"
    model = load_model(model_file)

    similarity_measure = MS2DeepScoreMonteCarlo(model, n_ensembles,
                                                average_type)
    return spectrums, model, similarity_measure


def test_MS2DeepScoreMonteCarlo_embedding_array(ms2_deep_score_instance):
    """Test embedding creation.
    """
    spectrums, _, similarity_measure = ms2_deep_score_instance

    x1 = similarity_measure.get_embedding_array(spectrums[:3])
    x2 = similarity_measure.get_embedding_array(spectrums[:3])
    assert x1.shape == x2.shape == (3, 100), "Expected different vector shape"
    assert (x1[0] != x2[0]).any(), "The embeddings should always look different"
    assert isinstance(x1, np.ndarray), "Expected embeddings to be numpy array"


def test_MS2DeepScoreMonteCarlo_embedding_ensemble(ms2_deep_score_instance):
    """Test embedding creation.
    """
    spectrums, _, similarity_measure = ms2_deep_score_instance

    ensemble = similarity_measure.get_embedding_ensemble(spectrums[0])
    assert ensemble.shape == (5, 100), "Expected different shape"
    assert isinstance(ensemble, np.ndarray), "Expected embeddings to be numpy array"


def test_MS2DeepScoreMonteCarlo_pair(ms2_deep_score_instance):
    """Test pair prediction.
    """
    spectrums, _, similarity_measure = ms2_deep_score_instance

    score = similarity_measure.pair(spectrums[0], spectrums[0])
    assert isinstance(score, np.ndarray), "Expected score to be numpy array"
    assert score["score"].size == 1
    assert score["lower_bound"].size == score["upper_bound"].size == 1
    assert score["score"].dtype == np.float32, "Expected float as score."
    assert score["lower_bound"].dtype == np.float32, "Expected float as uncertainty."


def test_MS2DeepScoreMonteCarlo_matrix(ms2_deep_score_instance):
    """Test matrix prediction.
    """
    spectrums, _, similarity_measure = ms2_deep_score_instance

    scores_array = similarity_measure.matrix(spectrums[:4], spectrums[:4])
    assert isinstance(scores_array, np.ndarray), "Expected scores to be numpy array"
    assert scores_array["score"].shape == (4, 4)
    assert scores_array["lower_bound"].shape == (4, 4)
    assert np.min(scores_array['lower_bound']) <= np.min(scores_array['score'])
    assert scores_array["upper_bound"].shape == (4, 4)
    assert np.min(scores_array['upper_bound']) >= np.min(scores_array['score'])
