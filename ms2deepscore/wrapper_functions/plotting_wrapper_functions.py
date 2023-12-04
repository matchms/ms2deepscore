import os
from typing import List
import numpy as np
from matchms import Spectrum
from matplotlib import pyplot as plt

from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from ms2deepscore.utils import load_pickled_file, save_pickled_file
from ms2deepscore.visualize_results.calculate_tanimoto_scores import get_tanimoto_score_between_spectra
from ms2deepscore.visualize_results.plotting import plot_histograms
from ms2deepscore.wrapper_functions.StoreTrainingData import StoreTrainingData
from ms2deepscore.train_new_model.SettingMS2Deepscore import SettingsMS2Deepscore


def benchmark_wrapper(val_spectra_1: List[Spectrum],
                      val_spectra_2: List[Spectrum],
                      benchmarking_results_folder: str,
                      ms2ds_model: MS2DeepScore,
                      file_name_prefix):
    true_values = create_or_load_true_values(val_spectra_1, val_spectra_2,
                                             benchmarking_results_folder,
                                             file_name_prefix)
    predictions = create_or_load_predicted_values(val_spectra_1, val_spectra_2,
                                                  benchmarking_results_folder,
                                                  ms2ds_model,
                                                  file_name_prefix)
    create_all_plots(predictions, true_values, benchmarking_results_folder, file_name_prefix)


def create_or_load_true_values(val_spectra_1: List[Spectrum],
                               val_spectra_2: List[Spectrum],
                               benchmarking_results_folder: str,
                               file_name_prefix):
    true_values_file_name = os.path.join(benchmarking_results_folder, f"{file_name_prefix}_true_values.pickle")
    if os.path.exists(true_values_file_name):
        true_values = load_pickled_file(true_values_file_name)
        print(f"Loaded in previous true values for: {file_name_prefix}")
    else:
        # Calculate true values
        true_values = get_tanimoto_score_between_spectra(val_spectra_1, val_spectra_2)
        save_pickled_file(true_values, true_values_file_name)
    return true_values


def create_or_load_predicted_values(val_spectra_1: List[Spectrum],
                                    val_spectra_2: List[Spectrum],
                                    benchmarking_results_folder: str,
                                    ms2ds_model: MS2DeepScore,
                                    file_name_prefix):
    predictions_file_name = os.path.join(benchmarking_results_folder, f"{file_name_prefix}_predictions.pickle")
    if os.path.exists(predictions_file_name):
        predictions = load_pickled_file(predictions_file_name)
        print(f"Loaded in previous predictions for: {file_name_prefix}")
    else:
        is_symmetric = (val_spectra_1 == val_spectra_2)
        predictions = ms2ds_model.matrix(val_spectra_1, val_spectra_2, is_symmetric=is_symmetric)
        save_pickled_file(predictions, predictions_file_name)
    return predictions


def create_all_plots(predictions,
                     true_values,
                     benchmarking_results_folder: str,
                     file_name_prefix):
    """Creates and saves plots and in between files for validation spectra

    Arguments:
        predictions: The predictions made by the model
        true_values: The true values predicted by the model
        benchmarking_results_folder: The results folder for the benchmarking
        ms2ds_model: The loaded ms2deepscore model
        file_name_prefix: The prefix used to create all file names.
    """
    plots_folder = os.path.join(benchmarking_results_folder, "plots_normalized_auc")
    os.makedirs(plots_folder, exist_ok=True)

    # Create plots
    plot_histograms(predictions, true_values, 10, 100)
    plt.savefig(os.path.join(plots_folder, f"{file_name_prefix}_plot.svg"))
    # Create reverse plot
    plot_histograms(true_values, predictions, 10, 100)
    plt.savefig(os.path.join(plots_folder, f"{file_name_prefix}_reversed_plot.svg"))

    mae = np.abs(predictions - true_values).mean()
    rmse = np.sqrt(np.square(predictions - true_values).mean())
    summary = f"For {file_name_prefix} the mae={mae} and rmse={rmse}\n"

    averages_summary_file = os.path.join(benchmarking_results_folder, "RMSE_and_MAE.txt")
    with open(averages_summary_file, "a", encoding="utf-8") as f:
        f.write(summary)
    print(summary)


def create_all_plots_wrapper(training_data_storing: StoreTrainingData,
                             settings: SettingsMS2Deepscore):
    _, positive_validation_spectra, _ = training_data_storing.load_positive_train_split()
    _, negative_validation_spectra, _ = training_data_storing.load_negative_train_split()

    model_folder = os.path.join(training_data_storing.trained_models_folder, settings.model_directory_name)

    # Check if the model already finished training
    if not os.path.exists(os.path.join(model_folder, settings.history_file_name)):
        raise ValueError(f"Did not plot since {model_folder} did not yet finish training")

    # Load in MS2Deepscore model
    ms2deepscore_model = MS2DeepScore(load_model(os.path.join(model_folder, settings.model_file_name)))

    # Create benchmarking results folder
    benchmarking_results_folder = os.path.join(model_folder, "benchmarking_results")
    os.makedirs(benchmarking_results_folder, exist_ok=True)

    benchmark_wrapper(positive_validation_spectra, positive_validation_spectra, benchmarking_results_folder,
                      ms2deepscore_model, "positive_positive")
    benchmark_wrapper(negative_validation_spectra, negative_validation_spectra, benchmarking_results_folder,
                      ms2deepscore_model, "negative_negative")
    benchmark_wrapper(negative_validation_spectra, positive_validation_spectra, benchmarking_results_folder,
                      ms2deepscore_model, "negative_positive")
    benchmark_wrapper(positive_validation_spectra + negative_validation_spectra,
                      positive_validation_spectra + negative_validation_spectra, benchmarking_results_folder,
                      ms2deepscore_model, "both_both")
