import os
from pathlib import Path
import numpy as np
from tensorflow import keras

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel, load_model
from tests.test_user_worfklow import load_processed_spectrums, get_reference_scores

TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def get_test_binner_and_generator():
    """Load test data and create instance of SpectrumBinner and data generator."""
    # Get test data
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    spectrum_binner = SpectrumBinner(400, mz_min=10.0, mz_max=500.0, peak_scaling=0.5)
    binned_spectrums = spectrum_binner.fit_transform(spectrums)

    dimension = len(spectrum_binner.known_bins)
    same_prob_bins = [(0, 0.5), (0.5, 1)]
    selected_inchikeys = tanimoto_scores_df.index[:60]

    # Create generator
    return spectrum_binner, \
        DataGeneratorAllInchikeys(binned_spectrums=binned_spectrums,
                                  selected_inchikeys=selected_inchikeys,
                                  reference_scores_df=tanimoto_scores_df,
                                  dim=dimension, same_prob_bins=same_prob_bins)


def test_siamese_model():
    spectrum_binner, test_generator = get_test_binner_and_generator()
    model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200),
                         embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.summary()
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)
    assert len(model.model.layers) == 4, "Expected different number of layers"
    assert len(model.model.layers[2].layers) == len(model.base.layers) == 11, \
        "Expected different number of layers"
    assert model.model.input_shape == [(None, 339), (None, 339)], "Expected different input shape"

    # Test base model inference
    X, y = test_generator.__getitem__(0)
    embeddings = model.base.predict(X[0])
    assert isinstance(embeddings, np.ndarray), "Expected numpy array"
    assert embeddings.shape[0] == test_generator.settings["batch_size"] == 32, \
        "Expected different batch size"
    assert embeddings.shape[1] == model.base.output_shape[1] == 200, \
        "Expected different embedding size"


def test_load_model():
    """Test loading a model from file."""
    spectrum_binner, test_generator = get_test_binner_and_generator()

    model_file = TEST_RESOURCES_PATH / "testmodel.hdf5"
    model = load_model(model_file)
    assert model.spectrum_binner.__dict__ == spectrum_binner.__dict__, "Expected different spectrum binner"

    # Test model layer shapes
    assert model.model.layers[2].to_json() == model.base.to_json(), \
        "Expected based model to be identical to part of main model."

    # Test base model inference
    X, y = test_generator.__getitem__(0)
    embeddings = model.base.predict(X[0])
    assert isinstance(embeddings, np.ndarray), "Expected numpy array"
    assert embeddings.shape[0] == test_generator.settings["batch_size"] == 32, \
        "Expected different batch size"
    assert embeddings.shape[1] == model.base.output_shape[1] == 200, \
        "Expected different embedding size"


def test_save_and_load_model(tmp_path):
    """Test saving and loading a model."""
    spectrum_binner, test_generator = get_test_binner_and_generator()
    model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200),
                         embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.summary()
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)

    # Write to test file
    filename = os.path.join(tmp_path, "model_export_test.hdf5")
    model.save(filename)

    # Test if file exists
    assert os.path.isfile(filename)

    # Test if content is correct
    model_import = load_model(filename)
    weights_original = model.base.layers[1].get_weights()[0]
    weights_imported = model_import.base.layers[1].get_weights()[0]
    assert np.all(weights_original == weights_imported), \
        "Imported and original model weights should be the same"
    assert model.model.to_json() == model_import.model.to_json(), \
        "Expect same architecture for original and imported model"
