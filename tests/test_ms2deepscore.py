import os
import numpy as np
import pytest
from tensorflow import keras

from ms2deepscore import MS2DeepScore
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel
# from tests.test_data_generators import create_test_data
# from tests.test_models import get_test_generator
from tests.test_user_worfklow import load_processed_spectrums, get_reference_scores


def get_test_ms2_deep_score_instance():
    """Test basic scoring using MS2DeepScore.
    TODO: adapt once model loading/saving is implemented properly!
    """
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    ms2ds_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)

    # Create generator
    dimension = len(ms2ds_binner.known_bins)
    test_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                               dim=dimension)
    # Train model
    model = SiameseModel(input_dim=dimension, base_dims=(200, 200, 200),
                         embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)

    similarity_measure = MS2DeepScore(model, ms2ds_binner)
    return spectrums, similarity_measure


def test_MS2DeepScore_vector_creation():
    """Test vector creation.
    """
    spectrums, similarity_measure = get_test_ms2_deep_score_instance()
    binned_spectrum0 = similarity_measure.spectrum_binner.transform([spectrums[0]])[0]
    input_vectors = similarity_measure._create_input_vector(binned_spectrum0)
    assert input_vectors.shape == (1, 543), "Expected different vector shape"
    assert isinstance(input_vectors, np.ndarray), "Expected vector to be numpy array"
    assert input_vectors[0, 114] == 1, "Expected different entries"


def test_MS2DeepScore_score_pair():
    """Test score calculation using *.pair* method.
    TODO: switch to pretrained model once possible
    """
    spectrums, similarity_measure = get_test_ms2_deep_score_instance()
    score = similarity_measure.pair(spectrums[0], spectrums[1])
    assert 0 < score < 1, "Expected score > 0 and < 1"
    assert isinstance(score, float), "Expected score to be float"


def test_MS2DeepScore_score_matrix():
    """Test score calculation using *.matrix* method.
    TODO: switch to pretrained model once possible
    """
    spectrums, similarity_measure = get_test_ms2_deep_score_instance()
    scores = similarity_measure.matrix(spectrums[:5], spectrums[:5])
    assert scores.shape == (5, 5), "Expected different score array shape"
    assert np.allclose([scores[i, i] for i in range(5)], 1.0), "Expected diagonal values to be approx 1.0"
