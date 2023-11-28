import os
from typing import List
import numpy as np
from matchms import Spectrum
from matplotlib import pyplot as plt
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from ms2deepscore.visualize_results.plotting import plot_histograms
from ms2deepscore.train_new_model.tanimoto_score_calculation import (
    calculate_tanimoto_scores_unique_inchikey,
    get_tanimoto_score_between_spectra)
from ms2deepscore.utils import (create_dir_if_missing, load_pickled_file,
                                save_pickled_file)


def create_all_plots(data_dir,
                     model_dir_name):
    positive_validation_spectra = load_pickled_file(os.path.join(data_dir, "training_and_validation_split",
                                                                 "positive_validation_spectra.pickle"))
    negative_validation_spectra = load_pickled_file(os.path.join(data_dir, "training_and_validation_split",
                                                                 "negative_validation_spectra.pickle"))

    model_folder = os.path.join(data_dir, "trained_models", model_dir_name)
    # Check if the model already finished training
    if not os.path.exists(os.path.join(model_folder, "history.txt")):
        print(f"Did not plot since {model_folder} did not yet finish training")

    # Create benchmarking results folder
    benchmarking_results_folder = os.path.join(model_folder, "benchmarking_results")
    create_dir_if_missing(benchmarking_results_folder)

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


def benchmark_wrapper(val_spectra_1: List[Spectrum],
                      val_spectra_2: List[Spectrum],
                      benchmarking_results_folder: str,
                      ms2ds_model: MS2DeepScore,
                      file_name_prefix):
    """Creates and saves plots and in between files for validation spectra

    Arguments:
        val_spectra_1: The first validation spectra
        val_spectra_2: The second validation spectra
        benchmarking_results_folder: The results folder for the benchmarking
        ms2ds_model: The loaded ms2deepscore model
        file_name_prefix: The prefix used to create all file names.
    """
    # pylint: disable=too-many-locals
    # Create predictions
    predictions_file_name = os.path.join(benchmarking_results_folder, f"{file_name_prefix}_predictions.pickle")
    if os.path.exists(predictions_file_name):
        predictions = load_pickled_file(predictions_file_name)
        print(f"Loaded in previous predictions for: {file_name_prefix}")
    else:
        is_symmetric = (val_spectra_1 == val_spectra_2)
        predictions = ms2ds_model.matrix(val_spectra_1, val_spectra_2, is_symmetric=is_symmetric)
        save_pickled_file(predictions, predictions_file_name)

    # Calculate true values
    true_values_file_name = os.path.join(benchmarking_results_folder, f"{file_name_prefix}_true_values.pickle")
    if os.path.exists(true_values_file_name):
        true_values = load_pickled_file(true_values_file_name)
        print(f"Loaded in previous true values for: {file_name_prefix}")
    else:
        tanimoto_df = calculate_tanimoto_scores_unique_inchikey(val_spectra_1, val_spectra_2)
        true_values = get_tanimoto_score_between_spectra(tanimoto_df, val_spectra_1, val_spectra_2)
        save_pickled_file(true_values, true_values_file_name)

    plots_folder = os.path.join(benchmarking_results_folder, "plots_normalized_auc")
    create_dir_if_missing(plots_folder)
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
