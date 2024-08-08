from pathlib import Path
import numpy as np
import pytest
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from tests.test_user_worfklow import load_processed_spectrums


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def get_test_ms2deepscore_instance():
    """Load data and models for MS2DeepScore unit tests."""
    spectrums = load_processed_spectrums()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.pt"
    model = load_model(model_file)

    similarity_measure = MS2DeepScore(model)
    return spectrums, model, similarity_measure


def test_MS2DeepScore_vector_creation():
    """Test embeddings creation.
    """
    spectrums, _, similarity_measure = get_test_ms2deepscore_instance()
    embeddings = similarity_measure.get_embedding_array(spectrums)
    assert embeddings.shape == (76, 100), "Expected different embeddings shape"
    assert isinstance(embeddings, np.ndarray), "Expected embeddings to be numpy array"


def test_MS2DeepScore_score_pair():
    """Test score calculation using *.pair* method."""
    spectrums, _, similarity_measure = get_test_ms2deepscore_instance()
    score = similarity_measure.pair(spectrums[0], spectrums[1])
    assert np.allclose(score, 0.990366, atol=1e-6), "Expected different score."
    assert isinstance(score, float), "Expected score to be float"


def test_MS2DeepScore_score_matrix():
    """Test score calculation using *.matrix* method."""
    spectrums, _, similarity_measure = get_test_ms2deepscore_instance()
    scores = similarity_measure.matrix(spectrums[:4], spectrums[:3])

    expected_scores = np.array([
        [1.        , 0.99036639, 0.99084978],
        [0.99036639, 1.        , 0.99399306],
        [0.99084978, 0.99399306, 1.        ],
        [0.98811793, 0.96436209, 0.97351075]
        ])
    assert np.allclose(expected_scores, scores, atol=1e-6), "Expected different scores."


def test_MS2DeepScore_score_matrix_symmetric():
    """Test score calculation using *.matrix* method."""
    spectrums, _, similarity_measure = get_test_ms2deepscore_instance()
    scores = similarity_measure.matrix(spectrums[:4], spectrums[:4], is_symmetric=True)
    expected_scores = np.array([
        [1.        , 0.99036639, 0.99084978, 0.98811793],
        [0.99036639, 1.        , 0.99399306, 0.96436209],
        [0.99084978, 0.99399306, 1.        , 0.97351075],
        [0.98811793, 0.96436209, 0.97351075, 1.        ]])
    assert np.allclose(expected_scores, scores, atol=1e-6), "Expected different scores."


# def test_MS2DeepScore_score_matrix_symmetric_wrong_use():
#     """Test if *.matrix* method gives correct exception."""
#     spectrums, _, similarity_measure = get_test_ms2_deep_score_instance()
#     expected_msg = "Expected references to be equal to queries for is_symmetric=True"
#     with pytest.raises(AssertionError) as msg:
#         _ = similarity_measure.matrix(spectrums[:4],
#                                       [spectrums[i] for i in [1,2,3,0]],
#                                       is_symmetric=True)
#     assert expected_msg in str(msg), "Expected different exception message"
#     assert not similarity_measure.multi_inputs

# def get_test_ms2_deep_score_instance_additional_inputs():
#     """Load data and models for MS2DeepScore unit tests."""
#     spectrums = load_processed_spectrums()
#
#     # Load pretrained model
#     model_file = TEST_RESOURCES_PATH / "testmodel_additional_input.hdf5"
#     model = load_model(model_file)
#
#     similarity_measure = MS2DeepScore(model)
#     return spectrums, model, similarity_measure

# def test_MS2DeepScore_score_additional_input_feature():
#     """Test vector creation."""
#     spectrums, model, similarity_measure = get_test_ms2_deep_score_instance_additional_inputs()
#
#     binned_spectrum0 = model.spectrum_binner.transform([spectrums[0]])[0]
#     inputs = similarity_measure._create_input_vector(binned_spectrum0)
#     assert isinstance(inputs, list), "Expected inputs to be list"
#     assert inputs[0].shape == (1, 543), "Expected different vector shape"
#     assert inputs[1].shape == (1, model.nr_of_additional_inputs), "Expected different shape for additional_input"
#     assert isinstance(inputs[0], np.ndarray), "Expected vector to be numpy array"
#     assert inputs[0][0, 92] == 0.0, "Expected different entries"
#     assert similarity_measure.multi_inputs
