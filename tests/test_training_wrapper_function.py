import os
from matchms.exporting import save_as_mgf
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.wrapper_functions.training_wrapper_functions import (
    StoreTrainingData, train_ms2deepscore_wrapper)
from tests.create_test_spectra import pesticides_test_spectra


def test_train_wrapper_ms2ds_model(tmp_path):
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:35]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[35:]]

    spectra_file_name = os.path.join(tmp_path, "clean_spectra.mgf")
    save_as_mgf(positive_mode_spectra+negative_mode_spectra,
                filename=spectra_file_name)
    settings = SettingsMS2Deepscore({"epochs": 2,
                                     "average_pairs_per_bin": 2,
                                     "ionisation_mode": "negative",
                                     "batch_size": 2})
    train_ms2deepscore_wrapper(spectra_file_name, settings, 5)
    expected_file_names = StoreTrainingData(spectra_file_name)
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, settings.model_file_name))
    assert os.path.isfile(expected_file_names.negative_mode_spectra_file)
    assert os.path.isfile(expected_file_names.negative_validation_spectra_file)
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, "benchmarking_results",
                                       "both_both_predictions.pickle"))
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, "benchmarking_results",
                                       "plots_1_spectrum_per_inchikey", "both_vs_both_stacked_histogram.svg"))
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, "binned_spectra",
                                       "binned_training_spectra.pickle"))
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, "settings.json"))
