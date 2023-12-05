import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from packaging import version
from tensorflow import keras


if version.parse(tf.__version__) >= version.parse("2.11"):
    AdamOptimizer = keras.optimizers.legacy.Adam
else:
    AdamOptimizer = keras.optimizers.Adam
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import (DataGeneratorAllInchikeys,
                                          DataGeneratorAllSpectrums)
from ms2deepscore.MetadataFeatureGenerator import StandardScaler
from ms2deepscore.models import SiameseModel, load_model
from tests.test_user_worfklow import (get_reference_scores,
                                      load_processed_spectrums)


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
                                  spectrum_binner=spectrum_binner, same_prob_bins=same_prob_bins)


def test_siamese_model():
    spectrum_binner, test_generator = get_test_binner_and_generator()
    model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200), embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=AdamOptimizer(learning_rate=0.001))
    model.summary()
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)
    assert len(model.model.layers) == 4, "Expected different number of layers"
    assert len(model.model.layers[2].layers) == len(model.base.layers) == 10, \
        "Expected different number of layers"
    assert model.model.input_shape == [(None, 339), (None, 339)], "Expected different input shape"
    np.testing.assert_array_almost_equal(model.base.layers[1].kernel_regularizer.l1, 1e-6), \
        "Expected different L1 regularization rate"
    np.testing.assert_array_almost_equal(model.base.layers[1].kernel_regularizer.l2, 1e-6), \
        "Expected different L2 regularization rate"

    # Test base model inference
    X, y = test_generator.__getitem__(0)
    embeddings = model.base.predict(X[0])
    assert isinstance(embeddings, np.ndarray), "Expected numpy array"
    assert embeddings.shape[0] == test_generator.settings.batch_size == 32, \
        "Expected different batch size"
    assert embeddings.shape[1] == model.base.output_shape[1] == 200, \
        "Expected different embedding size"


def test_siamese_model_different_architecture():
    spectrum_binner, test_generator = get_test_binner_and_generator()
    model = SiameseModel(spectrum_binner, base_dims=(200, 200, 100, 100, 100), embedding_dim=100, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=AdamOptimizer(learning_rate=0.001))
    assert len(model.model.layers) == 4, "Expected different number of layers"
    assert len(model.model.layers[2].layers) == len(model.base.layers) == 16, \
        "Expected different number of layers"
    assert model.model.input_shape == [(None, 339), (None, 339)], "Expected different input shape"
    assert model.base.output_shape == (None, 100), "Expected different output shape of base model"


def test_siamese_model_dropout_in_first_layer():
    spectrum_binner, test_generator = get_test_binner_and_generator()
    model = SiameseModel(spectrum_binner, base_dims=(200, 200, 100, 100, 100), embedding_dim=100, dropout_rate=0.2,
                         dropout_in_first_layer=True)
    model.compile(loss='mse', optimizer=AdamOptimizer(learning_rate=0.001))
    assert len(model.model.layers) == 4, "Expected different number of layers"
    assert len(model.model.layers[2].layers) == len(model.base.layers) == 17, \
        "Expected different number of layers"
    assert model.model.input_shape == [(None, 339), (None, 339)], "Expected different input shape"
    assert model.base.output_shape == (None, 100), "Expected different output shape of base model"


def test_siamese_model_different_regularization_rates():
    spectrum_binner, test_generator = get_test_binner_and_generator()
    model = SiameseModel(spectrum_binner, base_dims=(200,), embedding_dim=100, l1_reg=1e-7, l2_reg=1e-5)
    np.testing.assert_array_almost_equal(model.base.layers[1].kernel_regularizer.l1, 1e-7), \
        "Expected different L1 regularization rate"
    np.testing.assert_array_almost_equal(model.base.layers[1].kernel_regularizer.l2, 1e-5), \
        "Expected different L2 regularization rate"


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
    assert embeddings.shape[0] == test_generator.settings.batch_size == 32, \
        "Expected different batch size"
    assert embeddings.shape[1] == model.base.output_shape[1] == 200, \
        "Expected different embedding size"


def test_save_and_load_model(tmp_path):
    """Test saving and loading a model."""
    spectrum_binner, test_generator = get_test_binner_and_generator()
    model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200), embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=AdamOptimizer(learning_rate=0.001))
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
    assert model.model.summary() == model_import.model.summary(), \
        "Expect same architecture for original and imported model"


def get_test_binner_and_generator_additional_inputs():
    """Load test data and create instance of SpectrumBinner and data generator."""
    # Get test data
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    additional_inputs=(StandardScaler("precursor_mz", mean=0, std=1000), StandardScaler("precursor_mz", mean=0, std=100), )
    spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                     additional_metadata=additional_inputs)
    binned_spectrums = spectrum_binner.fit_transform(spectrums)

    dimension = len(spectrum_binner.known_bins)
    data_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                               spectrum_binner=spectrum_binner)

    # Create generator
    return spectrum_binner, data_generator


def test_save_and_load_model_additional_inputs(tmp_path):
    """Test saving and loading a model."""
    spectrum_binner, test_generator = get_test_binner_and_generator_additional_inputs()
    # generic retrieval of the input shape of additional inputs
    input, _ = test_generator[0]

    spectrum_length = len(input[0][0])
    nr_of_additional_input = len(input[1][0])

    model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200), embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=AdamOptimizer(learning_rate=0.001))
    model.summary()
    
    assert model.base.layers[2].input_shape == [(None, spectrum_length), (None, nr_of_additional_input)], \
                                    "Concatenate Layer has a false input shape"
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)

    # Write to test file
    filename = os.path.join(tmp_path, "model_export_test_additional_inputs.hdf5")
    model.save(filename)

    # Test if file exists
    assert os.path.isfile(filename)

    # Test if content is correct
    model_import = load_model(filename)
    weights_original = model.base.layers[4].get_weights()[0]
    weights_imported = model_import.base.layers[4].get_weights()[0]
    assert np.all(weights_original == weights_imported), \
        "Imported and original model weights should be the same"
    assert model.model.summary() == model_import.model.summary(), \
        "Expect same architecture for original and imported model"
    assert model.spectrum_binner.additional_metadata == (StandardScaler("precursor_mz", mean=0, std=1000), StandardScaler("precursor_mz", mean=0, std=100), )
