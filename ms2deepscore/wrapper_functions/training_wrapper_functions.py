"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""

import os
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import \
    create_all_plots_wrapper
from ms2deepscore.wrapper_functions.StoreTrainingData import StoreTrainingData


def train_ms2deepscore_wrapper(spectra_file_name,
                               settings: SettingsMS2Deepscore,
                               validation_split_fraction=20
                               ):
    """Splits data, trains a ms2deepscore model, and does benchmarking."""
    stored_training_data = StoreTrainingData(spectra_file_name, validation_split_fraction)

    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra = stored_training_data.load_training_data(settings.ionisation_mode, "training")
    validation_spectra = stored_training_data.load_training_data(settings.ionisation_mode, "validation")

    # Train model
    train_ms2ds_model(training_spectra, validation_spectra,
                      stored_training_data.trained_models_folder,
                      settings)

    # Create performance plots for validation spectra
    positive_validation_spectra = stored_training_data.load_positive_train_split("validation")
    negative_validation_spectra = stored_training_data.load_negative_train_split("validation")

    create_all_plots_wrapper(positive_validation_spectra=positive_validation_spectra,
                             negative_validation_spectra=negative_validation_spectra,
                             model_folder=os.path.join(stored_training_data.trained_models_folder,
                                                       settings.model_directory_name),
                             settings=settings)
    # todo store the settings as well in the settings.model_directory_name
    return settings.model_directory_name
