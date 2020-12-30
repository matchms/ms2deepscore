import numpy as np
from tensorflow import keras

from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from tests.test_data_generators import create_test_data


def get_test_generator():
    # Get test data
    spectrums_binned, score_array, inchikey_score_mapping, inchikeys_all = create_test_data()

    # Define other parameters
    batch_size = 10
    num_turns = 1
    dimension = 101
    same_prob_bins = [(0, 0.5), (0.5, 1)]

    inchikey_ids = np.arange(0, 80)

    # Create generator
    test_generator = DataGeneratorAllInchikeys(spectrums_binned, score_array, inchikey_ids,
                                               inchikey_score_mapping, inchikeys_all,
                                               dim=dimension, batch_size=batch_size,
                                               num_turns=num_turns,
                                               shuffle=True, ignore_equal_pairs=True,
                                               same_prob_bins=same_prob_bins,
                                               augment_peak_removal_max=0.0,
                                               augment_peak_removal_intensity=0.0,
                                               augment_intensity=0.0)
    return test_generator


def test_siamese_model():
    test_generator = get_test_generator()
    model = SiameseModel(base_dims=(10, 10, 10), embedding_dim=10, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.fit_generator(generator=test_generator,
              validation_data=test_generator,
              epochs=2)
