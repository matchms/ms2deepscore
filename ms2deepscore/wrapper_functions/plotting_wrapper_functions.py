import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from ms2deepscore.utils import load_pickled_file, load_spectra_as_list

from ms2deepscore.benchmarking_models.plot_stacked_histogram import plot_stacked_histogram_plot_wrapper, \
    plot_reversed_stacked_histogram_plot
from ms2deepscore.benchmarking_models.select_spectrum_pairs_for_visualization import sample_spectra_multiple_times


def create_all_plots(predictions,
                     true_values,
                     val_spectra_1,
                     val_spectra_2,
                     benchmarking_results_folder,
                     file_name_prefix,
                     ):
    """Creates and saves plots and in between files for validation spectra

    Arguments:
        predictions: The predictions made by the model
        true_values: The true values predicted by the model
        benchmarking_results_file_names: Class storing all the default file names and folder structure
    """

    selected_predictions, selected_true_values = sample_spectra_multiple_times(val_spectra=val_spectra_1,
                                                                               val_spectra_other_mode=val_spectra_2,
                                                                               predicted_values=predictions,
                                                                               true_values=true_values,
                                                                               nr_of_sample_times=100)
    # Create plots
    plot_stacked_histogram_plot_wrapper(
        ms2deepscore_predictions=selected_predictions, tanimoto_scores=selected_true_values, n_bins=10,
        title=file_name_prefix.replace("_", ""))
    plt.savefig(os.path.join(benchmarking_results_folder, f"{file_name_prefix}_stacked_histogram.svg"))

    # Create reverse plot
    plot_reversed_stacked_histogram_plot(
        tanimoto_scores=selected_true_values, ms2deepscore_predictions=selected_predictions,
        title=file_name_prefix.replace("_", ""))
    plt.savefig(os.path.join(benchmarking_results_folder, f"{file_name_prefix}reversed_stacked_histogram.svg"))

    # todo add the RMSE plot.
    mae = np.abs(selected_predictions - selected_true_values).mean()
    rmse = np.sqrt(np.square(selected_predictions - selected_true_values).mean())
    summary = f"For {file_name_prefix} the mae={mae} and rmse={rmse}\n"

    # with open(benchmarking_results_file_names.averages_summary_file_name, "a", encoding="utf-8") as f:
    #     f.write(summary)
    print(summary)


def create_plots_between_all_ionmodes(model_directory,
                                      results_folder=None):
    spectra_folder = os.path.join(model_directory, "..", "..", "training_and_validation_split")
    if results_folder is None:
        results_folder = os.path.join(model_directory, "benchmarking_results", "plots")
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

    for ionmode_1, ionmode_2 in possible_comparisons:
        create_all_plots(predictions=load_pickled_file(os.path.join(model_directory,
                                                                    "benchmarking_results",
                                                                    f"{ionmode_1}_{ionmode_2}_predictions.pickle")),
                         true_values=load_pickled_file(os.path.join(model_directory,
                                                                    "benchmarking_results",
                                                                    f"{ionmode_1}_{ionmode_2}_true_values.pickle")),
                         val_spectra_1=validation_spectra[ionmode_1],
                         val_spectra_2=validation_spectra[ionmode_2],
                         benchmarking_results_folder=results_folder,
                         file_name_prefix=f"{ionmode_1}_vs_{ionmode_2}")
