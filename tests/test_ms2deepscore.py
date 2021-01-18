import os
import numpy as np
import pytest
from tensorflow import keras

from ms2deepscore import MS2DeepScore
from ms2deepscore.models import SiameseModel
from tests.test_data_generators import create_test_data
from tests.test_models import get_test_generator


def test_MS2DeepScore():
    """first drafted test
    TODO: switch to pretrained model to make scores reproducible.
    """
    test_generator = get_test_generator()
    model = SiameseModel(input_dim=101, base_dims=(200, 200, 200),
                         embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)

    binned_spectrums, _ = create_test_data()
    similarity_measure = MS2DeepScore(model)

    # Test vector creation
    input_vectors = similarity_measure._create_input_vectors(binned_spectrums[0])
    assert input_vectors.shape == (1, 101), "Expected different vector shape"
    assert isinstance(input_vectors, np.ndarray), "Expected vector to be numpy array"
    assert input_vectors[0, 38] == 1, "Expected different entries"

    # calculate similarities (pair)
    score = similarity_measure.pair(binned_spectrums[0], binned_spectrums[1])
    assert 0 < score < 1, "Expected score > 0 and < 1"
    assert isinstance(score, float), "Expected score to be float"

    # calculate similarities (matrix)
    scores = similarity_measure.matrix(binned_spectrums[:5], binned_spectrums[:5])
    assert scores.shape == (5, 5), "Expected different score array shape"
    assert np.allclose([scores[i, i] for i in range(5)], 1.0), "Expected diagonal values to be approx 1.0"
