import os
import sys
from typing import List
import numpy as np
from matchms.Spectrum import Spectrum
from matplotlib import pyplot as plt
from tqdm import tqdm
from ms2deepscore.benchmarking_models.plot_rmse_per_bin import (
    plot_rmse_per_bin, plot_rmse_per_bin_multiple_benchmarks)
from ms2deepscore.benchmarking_models.plot_stacked_histogram import (
    plot_reversed_stacked_histogram_plot, plot_stacked_histogram_plot_wrapper)
from ms2deepscore.benchmarking_models.select_spectrum_pairs_for_visualization import \
    sample_spectra_multiple_times
from ms2deepscore.utils import load_pickled_file, load_spectra_as_list


def create_plots_for_all_models(models_directory,
                                results_folder=None):
    """Creates the plots for all models in a directory

    models_directory:
        A folder with directories containing trained ms2deepscore models
    results_folder:
        The folder to store the plots, if None, it will be stored in the folders of the models under benchmarking/plots"""

    for data_processing_dir in os.listdir(models_directory):
        trained_models_dir = os.path.join(models_directory, data_processing_dir, "trained_models")
        if os.path.isdir(trained_models_dir):
            for model_dir_name in tqdm(os.listdir(trained_models_dir),
                                       desc=f"Creation plots for all models in {trained_models_dir}"):
                if results_folder is not None:
                    results_folder_per_model = os.path.join(results_folder, data_processing_dir, model_dir_name)
                else:
                    results_folder_per_model = None
                create_plots_between_all_ionmodes(os.path.join(trained_models_dir, model_dir_name), results_folder_per_model)


def create_plots_between_all_ionmodes(model_directory,
                                      results_folder=None):
    spectra_folder = os.path.join(model_directory, "..", "..", "training_and_validation_split")
    if results_folder is None:
        results_folder = os.path.join(model_directory, "benchmarking_results", "plots_1_spectrum_per_inchikey")
    os.makedirs(results_folder, exist_ok=True)
    positive_validation_spectra = load_spectra_as_list(os.path.join(spectra_folder, "positive_validation_spectra.mgf"))
    negative_validation_spectra = load_spectra_as_list(os.path.join(spectra_folder, "negative_validation_spectra.mgf"))
    both_validation_spectra = positive_validation_spectra + negative_validation_spectra

    validation_spectra = {"positive": positive_validation_spectra,
                          "negative": negative_validation_spectra,
                          "both": both_validation_spectra}

    possible_comparisons = (("positive", "positive"),
                            ("negative", "positive"),
                            ("negative", "negative"),
                            ("both", "both"))
    all_selected_true_values = []
    all_selected_predictions = []
    all_labels = []
    for ionmode_1, ionmode_2 in possible_comparisons:
        print(f"Creating plots for {ionmode_1} vs {ionmode_2}")
        selected_true_values, selected_predictions = create_all_plots(
            predictions=load_pickled_file(os.path.join(model_directory,
                                                       "benchmarking_results",
                                                       f"{ionmode_1}_{ionmode_2}_predictions.pickle")),
            true_values=load_pickled_file(os.path.join(model_directory,
                                                       "benchmarking_results",
                                                       f"{ionmode_1}_{ionmode_2}_true_values.pickle")),
            val_spectra_1=validation_spectra[ionmode_1],
            val_spectra_2=validation_spectra[ionmode_2],
            benchmarking_results_folder=results_folder,
            file_name_prefix=f"{ionmode_1}_vs_{ionmode_2}")
        all_selected_true_values.append(selected_true_values)
        all_selected_predictions.append(selected_predictions)
        all_labels.append(f"{ionmode_1} vs {ionmode_2}")
    plot_rmse_per_bin_multiple_benchmarks(list_of_predicted_scores=all_selected_predictions,
                                          list_of_true_values=all_selected_true_values,
                                          labels=all_labels)
    plt.savefig(os.path.join(results_folder, "RMSE_comparison.svg"))


def create_all_plots(predictions: np.array,
                     true_values: np.array,
                     val_spectra_1: List[Spectrum],
                     val_spectra_2: List[Spectrum],
                     benchmarking_results_folder,
                     file_name_prefix: str,
                     ):  # pylint: disable=too-many-arguments
    """Creates and saves plots and in between files for validation spectra

    predictions:
        A matrix with the predictions made by the model.
    true_values:
        The true values predicted by the model
    val_spectra_1:
        The validation spectra.
    val_spectra_2:
    benchmarking_results_folder:
    file_name_prefix:
        Used for title and legends in figures and for file names. (for instance positive vs negative)
    """
    nr_of_sample_times = 100
    selected_predictions, selected_true_values = sample_spectra_multiple_times(val_spectra=val_spectra_1,
                                                                               val_spectra_other_mode=val_spectra_2,
                                                                               predicted_values=predictions,
                                                                               true_values=true_values,
                                                                               nr_of_sample_times=nr_of_sample_times)
    # Create plots
    plot_stacked_histogram_plot_wrapper(
        ms2deepscore_predictions=selected_predictions, tanimoto_scores=selected_true_values, n_bins=10,
        title=file_name_prefix.replace("_", " "), ms2deepscore_nr_of_bin_correction=nr_of_sample_times)
    plt.savefig(os.path.join(benchmarking_results_folder, f"{file_name_prefix}_stacked_histogram.svg"))

    # Create reverse plot
    plot_reversed_stacked_histogram_plot(
        tanimoto_scores=selected_true_values, ms2deepscore_predictions=selected_predictions,
        title=file_name_prefix.replace("_", " "), ms2deepscore_nr_of_bin_correction=nr_of_sample_times)
    plt.savefig(os.path.join(benchmarking_results_folder, f"{file_name_prefix}_reversed_stacked_histogram.svg"))
    plot_rmse_per_bin(predicted_scores=selected_predictions,
                      true_scores=selected_true_values)
    plt.savefig(os.path.join(benchmarking_results_folder, f"{file_name_prefix}_RMSE_per_bin.svg"))
    return selected_true_values, selected_predictions


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Could not run, please provide the model folder")
        sys.exit(1)

    model_directory = sys.argv[1]
    if len(sys.argv) == 3:
        results_folder = sys.argv[2]
    else:
        results_folder = None

    create_plots_for_all_models(models_directory=model_directory,
                                results_folder=results_folder)
