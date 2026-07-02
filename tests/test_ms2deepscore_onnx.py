from pathlib import Path
import numpy as np
import pytest

from ms2deepscore import MS2DeepScoreONNX
from ms2deepscore.models import SiameseSpectralModelONNX
from tests.create_test_spectra import pesticides_test_spectra

TEST_RESOURCES_PATH = Path(__file__).parent / "resources"


def get_test_ms2deepscore_onnx_instance():
    """Load data and models for MS2DeepScoreONNX unit tests."""
    spectrums = pesticides_test_spectra()

    # Load pretrained ONNX model
    model_file = TEST_RESOURCES_PATH / "testmodel_v1.onnx"
    model = SiameseSpectralModelONNX(model_file, validate_settings=False)

    similarity_measure = MS2DeepScoreONNX(model)
    return spectrums, model, similarity_measure


def test_MS2DeepScoreONNX_vector_creation():
    """Test embeddings creation."""
    spectrums, _, similarity_measure = get_test_ms2deepscore_onnx_instance()
    embeddings = similarity_measure.get_embedding_array(spectrums)

    assert embeddings.shape == (76, 100), "Expected different embeddings shape"
    assert isinstance(embeddings, np.ndarray), "Expected embeddings to be numpy array"


def test_MS2DeepScoreONNX_score_pair():
    """Test score calculation using *.pair* method."""
    spectrums, _, similarity_measure = get_test_ms2deepscore_onnx_instance()
    score = similarity_measure.pair(spectrums[0], spectrums[1])

    assert np.allclose(score, 0.990366, atol=1e-6), "Expected different score."
    assert isinstance(score, float) or isinstance(score, np.floating), "Expected score to be float"


def test_MS2DeepScoreONNX_score_matrix():
    """Test score calculation using *.matrix* method."""
    spectrums, model, similarity_measure = get_test_ms2deepscore_onnx_instance()
    scores = similarity_measure.matrix(spectrums[:4], spectrums[:3])

    expected_scores = np.array(
        [
            [1.0, 0.99036639, 0.99084978],
            [0.99036639, 1.0, 0.99399306],
            [0.99084978, 0.99399306, 1.0],
            [0.98811793, 0.96436209, 0.97351075],
        ]
    )


    # GPU Inference with OpenVino will downcast to FP16 and lead to less precision
    #tolerance = 1e-4 if "OpenVINOExecutionProvider" in model.session.get_providers() else 1e-6
    assert np.allclose(expected_scores, scores, atol=1e-6), "Expected different scores."

def test_MS2DeepScoreONNX_score_matrix_symmetric():
    """Test score calculation using *.matrix* method with is_symmetric=True."""
    spectrums, model, similarity_measure = get_test_ms2deepscore_onnx_instance()
    scores = similarity_measure.matrix(spectrums[:4], spectrums[:4], is_symmetric=True)

    expected_scores = np.array(
        [
            [1.0, 0.99036639, 0.99084978, 0.98811793],
            [0.99036639, 1.0, 0.99399306, 0.96436209],
            [0.99084978, 0.99399306, 1.0, 0.97351075],
            [0.98811793, 0.96436209, 0.97351075, 1.0],
        ]
    )

    assert np.allclose(expected_scores, scores, atol=1e-6), "Expected different scores."


def test_MS2DeepScoreONNX_score_matrix_symmetric_wrong_use():
    """Test if *.matrix* method gives correct exception when references != queries."""
    spectrums, _, similarity_measure = get_test_ms2deepscore_onnx_instance()
    expected_msg = "Expected references to be equal to queries for is_symmetric=True"

    with pytest.raises(AssertionError) as msg:
        _ = similarity_measure.matrix(spectrums[:4], [spectrums[i] for i in [1, 2, 3, 0]], is_symmetric=True)

    assert expected_msg in str(msg.value), "Expected different exception message"
