import os
import numpy as np
from matchms.exporting import save_as_mgf
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.models import load_model
from ms2deepscore.wrapper_functions.training_wrapper_functions import (
    StoreTrainingData, train_ms2deepscore_wrapper, parameter_search)
from tests.create_test_spectra import pesticides_test_spectra


def test_train_wrapper_ms2ds_model(tmp_path):
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:35]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[35:]]

    spectra_file_name = os.path.join(tmp_path, "clean_spectra.mgf")
    save_as_mgf(positive_mode_spectra+negative_mode_spectra,
                filename=spectra_file_name)
    settings = SettingsMS2Deepscore(**{
        "epochs": 2,  # to speed up tests --> usually many more
        "ionisation_mode": "negative",
        "base_dims": (200, 200),  # to speed up tests --> usually larger
        "embedding_dim": 100,  # to speed up tests --> usually larger
        "same_prob_bins": np.array([(-0.01, 0.2), (0.2, 1.0)]),
        "average_inchikey_sampling_count": 8,
        "batch_size": 2,  # to speed up tests --> usually larger
        "random_seed": 42
        })

    model_directory_name = train_ms2deepscore_wrapper(spectra_file_name, settings, validation_split_fraction=5)
    expected_file_names = StoreTrainingData(spectra_file_name)
    # Test model is created and can be loaded
    model_file_name = os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       model_directory_name, settings.model_file_name)
    assert os.path.isfile(model_file_name)
    load_model(model_file_name)
    assert os.path.isfile(expected_file_names.negative_mode_spectra_file)
    assert os.path.isfile(expected_file_names.negative_validation_spectra_file)
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       model_directory_name, "benchmarking_results", "average_per_bin.svg"))
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       model_directory_name, "settings.json"))


def test_store_training_data_with_spectra_file(tmp_path):
    """Test loading and splitting spectra when a spectra file is provided."""
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:35]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[35:]]

    spectra_file_name = os.path.join(tmp_path, "clean_spectra.mgf")
    save_as_mgf(positive_mode_spectra + negative_mode_spectra, filename=spectra_file_name)

    stored_data = StoreTrainingData(spectra_file_name, split_fraction=20, random_seed=42)

    # Test loading positive and negative mode spectra
    positive_spectra = stored_data.load_positive_mode_spectra()
    negative_spectra = stored_data.load_negative_mode_spectra()

    assert len(positive_spectra) == 35
    assert len(negative_spectra) == 41
    assert os.path.isfile(stored_data.positive_mode_spectra_file)
    assert os.path.isfile(stored_data.negative_mode_spectra_file)


def test_parameter_search_wrapper(tmp_path):
    """Test the parameter search wrapper."""
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:35]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[35:]]

    spectra_file_name = os.path.join(tmp_path, "clean_spectra.mgf")
    save_as_mgf(positive_mode_spectra + negative_mode_spectra, filename=spectra_file_name)

    base_settings = SettingsMS2Deepscore(**{
        "epochs": 2,  # to speed up tests
        "ionisation_mode": "negative",
        "base_dims": (200, 200),
        "patience": 1,
        "embedding_dim": 100,
        "same_prob_bins": np.array([(-0.01, 0.2), (0.2, 1.0)]),
        "average_inchikey_sampling_count": 8,
        "batch_size": 2,
        "random_seed": 42
    })

    setting_variations = {
        "epochs": [1, 2],
        "batch_size": [2, 4]
    }

    results = parameter_search(
        spectra_file_path_or_dir=spectra_file_name,
        base_settings=base_settings,
        setting_variations=setting_variations,
        validation_split_fraction=5,
        path_checkpoint=os.path.join(tmp_path, "results_checkpoint.pkl")
    )

    assert len(results) == 4  # Two variations of epochs and two for batch size = 2 * 2
    for _, value in results.items():
        assert "params" in value
        assert "history" in value
        assert "losses" in value
