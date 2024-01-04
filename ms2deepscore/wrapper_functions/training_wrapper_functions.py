"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""

import os

from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from ms2deepscore.wrapper_functions.StoreTrainingData import StoreTrainingData
from ms2deepscore.benchmarking_models.calculate_scores_for_validation import \
    calculate_true_values_and_predictions_for_validation_spectra
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import create_plots_between_all_ionmodes


def train_ms2deepscore_wrapper(spectra_file_path,
                               settings: SettingsMS2Deepscore,
                               validation_split_fraction=20
                               ):
    """Splits data, trains a ms2deepscore model, and does benchmarking."""

    stored_training_data = StoreTrainingData(spectra_file_path,
                                             split_fraction=validation_split_fraction)

    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra = stored_training_data.load_training_data(settings.ionisation_mode, "training")
    validation_spectra = stored_training_data.load_training_data(settings.ionisation_mode, "validation")

    # Train model
    train_ms2ds_model(training_spectra, validation_spectra,
                      stored_training_data.trained_models_folder,
                      settings)

    # Create performance plots for validation spectra
    ms2deepsore_model_file_name = os.path.join(stored_training_data.trained_models_folder,
                                               settings.model_directory_name,
                                               settings.model_file_name)
    calculate_true_values_and_predictions_for_validation_spectra(
        stored_training_data,
        ms2deepsore_model_file_name=ms2deepsore_model_file_name,
        results_directory=os.path.join(stored_training_data.trained_models_folder,
                                       settings.model_directory_name, "benchmarking_results"))

    create_plots_between_all_ionmodes(model_directory=os.path.join(stored_training_data.trained_models_folder,
                                                                   settings.model_directory_name))
    return settings.model_directory_name


