from pathlib import Path
import numpy as np
import pytest
from ms2deepscore import MS2DeepScoreMonteCarlo
from ms2deepscore.models import load_model
from tests.test_user_worfklow import load_processed_spectrums


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def get_test_ms2_deep_score_instance(n_ensembles, average_type="median"):
    """Load data and models for MS2DeepScore unit tests."""
    spectrums = load_processed_spectrums()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.hdf5"
    model = load_model(model_file)

    similarity_measure = MS2DeepScoreMonteCarlo(model, n_ensembles,
                                                average_type)
    return spectrums, model, similarity_measure


def test_MS2DeepScoreMonteCarlo_vector_creation():
    """Test vector creation.
    """
    spectrums, model, similarity_measure = get_test_ms2_deep_score_instance(n_ensembles=5)
    binned_spectrum0 = model.spectrum_binner.transform([spectrums[0]])[0]
    input_vectors = similarity_measure._create_input_vector(binned_spectrum0)
    embeddings = similarity_measure.calculate_vectors([spectrums[0]])
    assert input_vectors.shape == (1, 339), "Expected different vector shape"
    assert embeddings.shape == (5, 200), "Expected different embeddings array shape"
    assert isinstance(input_vectors, np.ndarray), "Expected vector to be numpy array"
    assert isinstance(embeddings, np.ndarray), "Expected embeddings to be numpy array"
    assert input_vectors[0, 92] == 1, "Expected different entries"


@pytest.mark.parametrize("average_type", ['median', 'mean'])
def test_MS2DeepScoreMonteCarlo_score_pair(average_type):
    """Test score calculation using *.pair* method."""
    spectrums, _, similarity_measure = get_test_ms2_deep_score_instance(n_ensembles=5,
                                                                        average_type=average_type)
    score = similarity_measure.pair(spectrums[0], spectrums[1])
    assert score['score'].dtype == np.float64, "Expected float as score."
    assert score['score'] > 0.65 and score['score'] < 0.9, "Expected score in different range"
    assert score['uncertainty'].dtype == np.float64, "Expected float as uncertainty."
    if average_type == 'median':
        assert score['uncertainty'] > 0.01 and score['uncertainty'] < 0.12, \
            "Expected uncertainty in different range"
    else:
        assert score['uncertainty'] > 0.01 and score['uncertainty'] < 0.06, \
            "Expected uncertainty in different range"


@pytest.mark.parametrize("average_type", ['median', 'mean'])
def test_MS2DeepScoreMonteCarlo_score_matrix(average_type):
    """Test score calculation using *.matrix* method."""
    spectrums, _, similarity_measure = get_test_ms2_deep_score_instance(n_ensembles=5,
                                                                        average_type=average_type)
    scores = similarity_measure.matrix(spectrums[:4], spectrums[:3])
    assert scores['score'].shape == (4, 3), "Expected different shape"
    assert scores['uncertainty'].shape == (4, 3), "Expected different shape"
    assert np.max(scores['uncertainty']) < 0.2, "Expected lower uncertainty"
    assert np.max(scores['score']) > 0.5, "Expected higher scores"


def test_MS2DeepScoreMonteCarlo_score_matrix_symmetric_wrong_use():
    """Test if *.matrix* method gives correct exception."""
    spectrums, _, similarity_measure = get_test_ms2_deep_score_instance(n_ensembles=2)
    expected_msg = "Expected references to be equal to queries for is_symmetric=True"
    with pytest.raises(AssertionError) as msg:
        _ = similarity_measure.matrix(spectrums[:4],
                                      [spectrums[i] for i in [1,2,3,0]],
                                      is_symmetric=True)
    assert expected_msg in str(msg), "Expected different exception message"
