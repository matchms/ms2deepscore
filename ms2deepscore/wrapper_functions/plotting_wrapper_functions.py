import os
from typing import List
import numpy as np
from matchms import Spectrum
from matplotlib import pyplot as plt
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.utils import load_pickled_file, save_pickled_file
from ms2deepscore.visualize_results.calculate_tanimoto_scores import \
    get_tanimoto_score_between_spectra
from ms2deepscore.visualize_results.plot_stacked_histogram import plot_stacked_histogram_plot_wrapper, \
    plot_reversed_stacked_histogram_plot


class BenchmarkingResultsFileNames:
    def __init__(self, model_folder, file_name_prefix):
        self.file_name_prefix = file_name_prefix
        self.benchmarking_results_folder = os.path.join(model_folder, "benchmarking_results")
        os.makedirs(self.benchmarking_results_folder, exist_ok=True)
        self.true_values_file_name = os.path.join(self.benchmarking_results_folder, f"{file_name_prefix}_true_values.pickle")
        self.predictions_file_name = os.path.join(self.benchmarking_results_folder, f"{file_name_prefix}_predictions.pickle")

        self.plots_folder = os.path.join(self.benchmarking_results_folder, "plots")
        os.makedirs(self.plots_folder, exist_ok=True)

        self.normal_plot_file_name = os.path.join(self.plots_folder, f"{file_name_prefix}_plot.svg")
        self.reversed_plot_file_name = os.path.join(self.plots_folder, f"{file_name_prefix}_reversed_plot.svg")
        self.averages_summary_file_name = os.path.join(self.benchmarking_results_folder, "RMSE_and_MAE.txt")


def benchmark_wrapper(val_spectra_1: List[Spectrum],
                      val_spectra_2: List[Spectrum],
                      benchmarking_results_file_names: BenchmarkingResultsFileNames,
                      ms2ds_model: MS2DeepScore):
    true_values = create_or_load_true_values(val_spectra_1, val_spectra_2,
                                             benchmarking_results_file_names.true_values_file_name)
    predictions = create_or_load_predicted_values(val_spectra_1, val_spectra_2,
                                                  benchmarking_results_file_names.predictions_file_name,
                                                  ms2ds_model)
    create_all_plots(predictions, true_values, benchmarking_results_file_names)


def create_or_load_true_values(val_spectra_1: List[Spectrum],
                               val_spectra_2: List[Spectrum],
                               true_values_file_name):
    if os.path.exists(true_values_file_name):
        true_values = load_pickled_file(true_values_file_name)
        print(f"Loaded in previous true values from: {true_values_file_name}")
    else:
        # Calculate true values
        true_values = get_tanimoto_score_between_spectra(val_spectra_1, val_spectra_2)
        save_pickled_file(true_values, true_values_file_name)
    return true_values


def create_or_load_predicted_values(val_spectra_1: List[Spectrum],
                                    val_spectra_2: List[Spectrum],
                                    predictions_file_name: str,
                                    ms2ds_model: MS2DeepScore):
    if os.path.exists(predictions_file_name):
        predictions = load_pickled_file(predictions_file_name)
        print(f"Loaded in previous predictions from {predictions_file_name}")
    else:
        is_symmetric = (val_spectra_1 == val_spectra_2)
        predictions = ms2ds_model.matrix(val_spectra_1, val_spectra_2, is_symmetric=is_symmetric)
        save_pickled_file(predictions, predictions_file_name)
    return predictions


def create_all_plots(predictions,
                     true_values,
                     benchmarking_results_file_names: BenchmarkingResultsFileNames):
    """Creates and saves plots and in between files for validation spectra

    Arguments:
        predictions: The predictions made by the model
        true_values: The true values predicted by the model
        benchmarking_results_file_names: Class storing all the default file names and folder structure
    """
    # Create plots
    plot_stacked_histogram_plot_wrapper(ms2deepscore_predictions=predictions,
                                        tanimoto_scores=true_values, n_bins=10)
    plt.savefig(benchmarking_results_file_names.normal_plot_file_name)
    # Create reverse plot
    plot_reversed_stacked_histogram_plot(tanimoto_scores=true_values,
                                         ms2deepscore_predictions=predictions)
    plt.savefig(benchmarking_results_file_names.reversed_plot_file_name)

    mae = np.abs(predictions - true_values).mean()
    rmse = np.sqrt(np.square(predictions - true_values).mean())
    summary = f"For {benchmarking_results_file_names.file_name_prefix} the mae={mae} and rmse={rmse}\n"

    with open(benchmarking_results_file_names.averages_summary_file_name, "a", encoding="utf-8") as f:
        f.write(summary)
    print(summary)


def create_all_plots_wrapper(positive_validation_spectra,
                             negative_validation_spectra,
                             model_folder,
                             settings: SettingsMS2Deepscore):
    # Check if the model already finished training
    if not os.path.exists(os.path.join(model_folder, settings.history_file_name)):
        raise ValueError(f"Did not plot since {model_folder} did not yet finish training")

    # Load in MS2Deepscore model
    ms2deepscore_model = MS2DeepScore(load_model(os.path.join(model_folder, settings.model_file_name)))

    benchmark_wrapper(positive_validation_spectra, positive_validation_spectra,
                      BenchmarkingResultsFileNames(model_folder, "positive_positive"),
                      ms2deepscore_model)
    benchmark_wrapper(negative_validation_spectra, negative_validation_spectra,
                      BenchmarkingResultsFileNames(model_folder, "negative_negative"),
                      ms2deepscore_model, )
    benchmark_wrapper(negative_validation_spectra, positive_validation_spectra,
                      BenchmarkingResultsFileNames(model_folder, "negative_positive"),
                      ms2deepscore_model, )
    benchmark_wrapper(positive_validation_spectra + negative_validation_spectra,
                      positive_validation_spectra + negative_validation_spectra,
                      BenchmarkingResultsFileNames(model_folder, "both_both"),
                      ms2deepscore_model, )
