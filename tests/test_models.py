import numpy as np
from tensorflow import keras

from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from tests.test_data_generators import create_test_data


def get_test_generator():
    # Get test data
    binned_spectrums, tanimoto_scores_df = create_test_data()

    dimension = 101
    same_prob_bins = [(0, 0.5), (0.5, 1)]
    selected_inchikeys = tanimoto_scores_df.index[:80]

    # Create generator
    return DataGeneratorAllInchikeys(binned_spectrums=binned_spectrums,
                                     selected_inchikeys=selected_inchikeys,
                                     labels_df=tanimoto_scores_df,
                                     dim=dimension, same_prob_bins=same_prob_bins)


def test_siamese_model():
    test_generator = get_test_generator()
    model = SiameseModel(input_dim=101, base_dims=(200, 200, 200),
                         embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.summary()
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)
    assert len(model.model.layers) == 4, "Expected different number of layers"
    assert len(model.model.layers[2].layers) == len(model.base.layers) == 11, \
        "Expected different number of layers"
    assert model.model.input_shape == [(None, 101), (None, 101)], "Expected different input shape"

    # Test base model inference
    X, y = test_generator.__getitem__(0)
    embeddings = model.base.predict(X[0])
    assert isinstance(embeddings, np.ndarray), "Expected numpy array"
    assert embeddings.shape[0] == test_generator.settings["batch_size"] == 32, \
        "Expected different batch size"
    assert embeddings.shape[1] == model.base.output_shape[1] == 200, \
        "Expected different embedding size"
