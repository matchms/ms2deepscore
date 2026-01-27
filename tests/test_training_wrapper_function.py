import os
import numpy as np
from matchms.exporting import save_as_mgf
from matchms.importing import load_spectra

from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore, SettingsEmbeddingEvaluator
from ms2deepscore.models import load_model, load_embedding_evaluator
from ms2deepscore.wrapper_functions.training_wrapper_functions import (train_ms2deepscore_wrapper, parameter_search,
                                                                       split_data_if_necessary)
from tests.create_test_spectra import pesticides_test_spectra


def test_train_wrapper_ms2ds_model(tmp_path):
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:35]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[35:]]

    spectra_file_name = os.path.join(tmp_path, "clean_spectra.mgf")
    save_as_mgf(positive_mode_spectra+negative_mode_spectra,
                filename=spectra_file_name)
    settings = SettingsMS2Deepscore(**{
        "spectrum_file_path": spectra_file_name,
        "epochs": 2,  # to speed up tests --> usually many more
        "ionisation_mode": "both",
        "base_dims": (200, 200),  # to speed up tests --> usually larger
        "embedding_dim": 100,  # to speed up tests --> usually larger
        "same_prob_bins": np.array([(-0.01, 0.2), (0.2, 1.0)]),
        "average_inchikey_sampling_count": 8,
        "batch_size": 2,  # to speed up tests --> usually larger
        "random_seed": 42,
        "train_test_split_fraction": 5,
        })

    model_directory_name = train_ms2deepscore_wrapper(settings, SettingsEmbeddingEvaluator())
    # Test model is created and can be loaded
    model_file_name = os.path.join(model_directory_name, settings.model_file_name)
    assert os.path.isfile(model_file_name)
    load_model(model_file_name)
    embedding_evaluator_file_name = os.path.join(settings.model_directory_name, "embedding_evaluator.pt")
    assert os.path.isfile(embedding_evaluator_file_name)
    load_embedding_evaluator(embedding_evaluator_file_name)
    assert os.path.isfile(os.path.join(tmp_path, settings.results_folder,
                                       model_directory_name, "benchmarking_results", "average_per_bin.svg"))
    assert os.path.isfile(os.path.join(tmp_path, settings.results_folder,
                                       model_directory_name, "settings.json"))


def test_store_training_data_with_spectra_file(tmp_path):
    """Test loading and splitting spectra when a spectra file is provided."""
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:35]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[35:]]

    spectra_file_name = os.path.join(tmp_path, "clean_spectra.mgf")
    save_as_mgf(positive_mode_spectra + negative_mode_spectra, filename=spectra_file_name)

    settings = SettingsMS2Deepscore(**{
        "spectrum_file_path": spectra_file_name,
        "random_seed": 42,
        "train_test_split_fraction": 20,
        })
    split_data_if_necessary(settings)

    assert os.path.isfile(settings.training_spectra_file_name)
    assert os.path.isfile(settings.validation_spectra_file_name)
    assert os.path.isfile(settings.test_spectra_file_name)
    assert len(list(load_spectra(settings.training_spectra_file_name))) > 0
    assert len(list(load_spectra(settings.validation_spectra_file_name))) > 0
    assert len(list(load_spectra(settings.test_spectra_file_name))) > 0


def test_parameter_search_wrapper(tmp_path):
    """Test the parameter search wrapper."""
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:35]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[35:]]

    spectra_file_name = os.path.join(tmp_path, "clean_spectra.mgf")
    save_as_mgf(positive_mode_spectra + negative_mode_spectra, filename=spectra_file_name)

    base_settings = SettingsMS2Deepscore(**{
        "spectrum_file_path": spectra_file_name,
        "epochs": 2,  # to speed up tests
        "ionisation_mode": "negative",
        "base_dims": (200, 200),
        "patience": 1,
        "embedding_dim": 100,
        "same_prob_bins": np.array([(-0.01, 0.2), (0.2, 1.0)]),
        "average_inchikey_sampling_count": 8,
        "batch_size": 2,
        "random_seed": 42,
        "train_test_split_fraction": 5,
    })

    setting_variations = {
        "epochs": [1, 2],
        "batch_size": [2, 4]
    }

    results = parameter_search(
        base_settings=base_settings,
        setting_variations=setting_variations,
        path_checkpoint=os.path.join(tmp_path, "results_checkpoint.pkl")
    )

    assert len(results) == 4  # Two variations of epochs and two for batch size = 2 * 2
    for _, value in results.items():
        assert "params" in value
        assert "history" in value
        assert "losses" in value
