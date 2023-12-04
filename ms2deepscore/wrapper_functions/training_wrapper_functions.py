"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""

import os

from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import \
    create_all_plots_wrapper
from ms2deepscore.wrapper_functions.StoreTrainingData import StoreTrainingData


def train_ms2deepscore_wrapper(data_directory,
                               spectra_file_name,
                               settings: SettingsMS2Deepscore,
                               validation_split_fraction=20
                               ):
    """Trains a ms2deepscore model, including the data split into pos,val,train spectra."""
    stored_training_data = StoreTrainingData(data_directory, spectra_file_name, validation_split_fraction)

    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra, validation_spectra, _ = stored_training_data.load_train_val_data(settings.ionisation_mode)

    # Train model
    train_ms2ds_model(training_spectra, validation_spectra,
                      stored_training_data.trained_models_folder,
                      settings)

    # Create performance plots for validation spectra
    _, positive_validation_spectra, _ = stored_training_data.load_positive_train_split()
    _, negative_validation_spectra, _ = stored_training_data.load_negative_train_split()

    create_all_plots_wrapper(positive_validation_spectra,
                             negative_validation_spectra,
                             model_folder=os.path.join(stored_training_data.trained_models_folder,
                                                       settings.model_directory_name),
                             settings=settings)
    # todo store the settings as well in the settings.model_directory_name
    return settings.model_directory_name
