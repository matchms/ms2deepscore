from pathlib import Path
import numpy as np
from ms2deepscore import MS2DeepScoreEvaluated
from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator
from ms2deepscore.models import load_model, LinearModel, EmbeddingEvaluationModel
from tests.create_test_spectra import pesticides_test_spectra


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def get_test_ms2deepscore_evaluated_instance():
    """Load data and models for MS2DeepScore unit tests."""
    spectrums = pesticides_test_spectra()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.pt"
    model = load_model(model_file)

    embedding_evaluator = EmbeddingEvaluationModel(SettingsEmbeddingEvaluator())
    score_evaluator = LinearModel(2)
    score_evaluator.fit(np.random.uniform(0, 0.5, (100, 2)), np.random.random((100)))

    similarity_measure = MS2DeepScoreEvaluated(model, embedding_evaluator, score_evaluator)
    return spectrums, similarity_measure


def test_MS2DeepScore_vector_creation():
    """Test embeddings creation.
    """
    spectrums, similarity_measure = get_test_ms2deepscore_evaluated_instance()
    embeddings = similarity_measure.get_embedding_array(spectrums)
    assert embeddings.shape == (76, 100), "Expected different embeddings shape"
    assert isinstance(embeddings, np.ndarray), "Expected embeddings to be numpy array"


def test_MS2DeepScore_score_pair():
    """Test score calculation using *.pair* method."""
    spectrums, similarity_measure = get_test_ms2deepscore_evaluated_instance()
    score = similarity_measure.pair(spectrums[0], spectrums[1])
    assert np.allclose(score["score"], 0.990366, atol=1e-6), "Expected different score."
    assert score["predicted_absolute_error"] > 0
    assert isinstance(score["score"], np.ndarray)
    assert isinstance(score["predicted_absolute_error"], np.ndarray)


def test_MS2DeepScore_score_matrix():
    """Test score calculation using *.matrix* method."""
    spectrums, similarity_measure = get_test_ms2deepscore_evaluated_instance()
    scores = similarity_measure.matrix(spectrums[:3], spectrums[:4])

    expected_scores = np.array([
        [1.        , 0.9903664 , 0.9908498 , 0.98811793],
        [0.9903664 , 1.        , 0.99399304, 0.9643621 ],
        [0.9908498 , 0.99399304, 1.        , 0.97351074]
        ])
    assert np.allclose(expected_scores, scores["score"], atol=1e-6), "Expected different scores."
    assert scores["predicted_absolute_error"].shape == (3, 4)
