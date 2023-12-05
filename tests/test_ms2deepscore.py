from pathlib import Path
import numpy as np
import pytest
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from tests.test_user_worfklow import load_processed_spectrums


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def get_test_ms2_deep_score_instance():
    """Load data and models for MS2DeepScore unit tests."""
    spectrums = load_processed_spectrums()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.hdf5"
    model = load_model(model_file)

    similarity_measure = MS2DeepScore(model)
    return spectrums, model, similarity_measure


def test_MS2DeepScore_vector_creation():
    """Test vector creation.
    """
    spectrums, model, similarity_measure = get_test_ms2_deep_score_instance()
    binned_spectrum0 = model.spectrum_binner.transform([spectrums[0]])[0]
    input_vectors = similarity_measure._create_input_vector(binned_spectrum0)
    assert input_vectors.shape == (1, 339), "Expected different vector shape"
    assert isinstance(input_vectors, np.ndarray), "Expected vector to be numpy array"
    assert input_vectors[0, 92] == 1, "Expected different entries"


def test_MS2DeepScore_score_pair():
    """Test score calculation using *.pair* method."""
    spectrums, _, similarity_measure = get_test_ms2_deep_score_instance()
    score = similarity_measure.pair(spectrums[0], spectrums[1])
    assert np.allclose(score, 0.92501721, atol=1e-6), "Expected different score."
    assert isinstance(score, float), "Expected score to be float"


def test_MS2DeepScore_score_matrix():
    """Test score calculation using *.matrix* method."""
    spectrums, _, similarity_measure = get_test_ms2_deep_score_instance()
    scores = similarity_measure.matrix(spectrums[:4], spectrums[:3])
    expected_scores = np.array([[1.        , 0.92501721, 0.8663899 ],
                                [0.92501721, 1.        , 0.86038138],
                                [0.8663899 , 0.86038138, 1.        ],
                                [0.91697757, 0.89758966, 0.79661344]])
    assert np.allclose(expected_scores, scores, atol=1e-6), "Expected different scores."


def test_MS2DeepScore_score_matrix_symmetric():
    """Test score calculation using *.matrix* method."""
    spectrums, _, similarity_measure = get_test_ms2_deep_score_instance()
    scores = similarity_measure.matrix(spectrums[:4], spectrums[:4], is_symmetric=True)
    expected_scores = np.array([[1.        , 0.92501721, 0.8663899 , 0.91697757],
                                [0.92501721, 1.        , 0.86038138, 0.89758966],
                                [0.8663899 , 0.86038138, 1.        , 0.79661344],
                                [0.91697757, 0.89758966, 0.79661344, 1.        ]])
    assert np.allclose(expected_scores, scores, atol=1e-6), "Expected different scores."


def test_MS2DeepScore_score_matrix_symmetric_wrong_use():
    """Test if *.matrix* method gives correct exception."""
    spectrums, _, similarity_measure = get_test_ms2_deep_score_instance()
    expected_msg = "Expected references to be equal to queries for is_symmetric=True"
    with pytest.raises(AssertionError) as msg:
        _ = similarity_measure.matrix(spectrums[:4],
                                      [spectrums[i] for i in [1,2,3,0]],
                                      is_symmetric=True)
    assert expected_msg in str(msg), "Expected different exception message"
    assert not similarity_measure.multi_inputs

def get_test_ms2_deep_score_instance_additional_inputs():
    """Load data and models for MS2DeepScore unit tests."""
    spectrums = load_processed_spectrums()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel_additional_input.hdf5"
    model = load_model(model_file)

    similarity_measure = MS2DeepScore(model)
    return spectrums, model, similarity_measure

def test_MS2DeepScore_score_additional_input_feature():
    """Test vector creation."""
    spectrums, model, similarity_measure = get_test_ms2_deep_score_instance_additional_inputs()

    binned_spectrum0 = model.spectrum_binner.transform([spectrums[0]])[0]
    inputs = similarity_measure._create_input_vector(binned_spectrum0)
    assert isinstance(inputs, list), "Expected inputs to be list"
    assert inputs[0].shape == (1, 543), "Expected different vector shape"
    assert inputs[1].shape == (1, model.nr_of_additional_inputs), "Expected different shape for additional_input"
    assert isinstance(inputs[0], np.ndarray), "Expected vector to be numpy array"
    assert inputs[0][0, 92] == 0.0, "Expected different entries"
    assert similarity_measure.multi_inputs
