"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""
import os

from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from ms2deepscore.visualize_results.plotting_wrapper_functions import benchmark_wrapper
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

    create_all_plots(stored_training_data,
                     settings.model_directory_name)
    # todo store the settings as well in the settings.model_directory_name
    return settings.model_directory_name


def create_all_plots(training_data_storing: StoreTrainingData,
                     model_dir_name):
    _, positive_validation_spectra, _ = \
        training_data_storing.load_positive_train_split()
    _, negative_validation_spectra, _ = \
        training_data_storing.load_negative_train_split()

    model_folder = os.path.join(training_data_storing.trained_models_folder,
                                model_dir_name)
    # Check if the model already finished training
    if not os.path.exists(os.path.join(model_folder, "history.txt")):
        print(f"Did not plot since {model_folder} did not yet finish training")

    # Create benchmarking results folder
    benchmarking_results_folder = os.path.join(model_folder, "benchmarking_results")
    os.makedirs(benchmarking_results_folder, exist_ok=True)

    # Load in MS2Deepscore model
    ms2deepscore_model = MS2DeepScore(load_model(os.path.join(model_folder, "ms2deepscore_model.hdf5")))

    benchmark_wrapper(positive_validation_spectra, positive_validation_spectra, benchmarking_results_folder,
                      ms2deepscore_model, "positive_positive")
    benchmark_wrapper(negative_validation_spectra, negative_validation_spectra, benchmarking_results_folder,
                      ms2deepscore_model, "negative_negative")
    benchmark_wrapper(negative_validation_spectra, positive_validation_spectra, benchmarking_results_folder,
                      ms2deepscore_model, "negative_positive")
    benchmark_wrapper(positive_validation_spectra + negative_validation_spectra,
                      positive_validation_spectra + negative_validation_spectra, benchmarking_results_folder,
                      ms2deepscore_model, "both_both")
