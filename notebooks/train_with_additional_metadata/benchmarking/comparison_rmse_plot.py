import os
import numpy as np
from typing import Dict, List
from matplotlib import pyplot as plt
from ms2deepscore.utils import load_pickled_file, return_non_existing_file_name
from ms2deepscore.plotting import create_histograms_plot, plot_rmse_per_bin, tanimoto_dependent_losses
from ms2deepscore.train_new_model.calculate_tanimoto_matrix import select_inchi_for_unique_inchikeys


def plot_multiple_rmses(rmse_dict: Dict[str, List[float]], bin_content: List[float], title, bounds,
                        plot_nr_of_spectrum_pairs):
    """Makes two plots the top plot has multiple lines and a legend the bottom one only has one

    This is used to make comparision plots of rmses per tanimoto score bin

    rmse_dict:
    Keys are the labels and values are a list with rmses
    bin_content is a li"""
    if plot_nr_of_spectrum_pairs:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10), dpi=120)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 7), dpi=120)
        ax1.set_xlabel("Tanimoto score bin")

    for label, rmses in rmse_dict.items():
        ax1.plot(np.arange(len(rmses)), rmses, "o:",
                 label=label)
    ax1.legend(loc='upper left')
    ax1.set_title('RMSE')
    ax1.set_ylabel("RMSE")
    ax1.grid(True)
    ax1.set_ylim([0.125, 0.55])
    if plot_nr_of_spectrum_pairs:
        ax2.plot(np.arange(len(bin_content)), bin_content, "o:", color="teal")
        ax2.set_title('# of spectrum pairs')
        ax2.set_ylabel("# of spectrum pairs")
        ax2.set_xlabel("Tanimoto score bin")
        ax2.grid(True)
        plt.yscale('log')
        ax2.legend(loc='upper left')
    fig.suptitle(title)
    plt.xticks(np.arange(len(rmses)),
               [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
    plt.tight_layout()


def plot_multiple_rms_per_bin(file_names_dict, validation_type, save_folder,
                              plot_nr_of_spectrum_pairs,
                              data_dir="../../../../data/",
                              ):

    rmses_per_file = {}
    for file_label, file_name in file_names_dict.items():
        benchmarking_results_folder = os.path.join(data_dir,
                                                   "trained_models",
                                                   file_name,
                                                   "benchmarking_results")
        predictions = load_pickled_file(os.path.join(benchmarking_results_folder,
                                                     f"{validation_type}_predictions.pickle"))
        true_values = load_pickled_file(os.path.join(benchmarking_results_folder,
                                                     f"{validation_type}_true_values.pickle"))

        ref_score_bins = np.linspace(0, 1.0, 11)
        bin_content, bounds, rmses, maes = tanimoto_dependent_losses(predictions,
                                                                     true_values, ref_score_bins)
        rmses_per_file[file_label] = rmses
    plot_multiple_rmses(rmses_per_file,
                        bin_content,
                        validation_type,
                        bounds,
                        plot_nr_of_spectrum_pairs)
    file_name = validation_type
    if plot_nr_of_spectrum_pairs:
        file_name += "with_spectrum_pairs"
    file_name += ".svg"
    plt.savefig(return_non_existing_file_name(os.path.join(save_folder, file_name)))


def plot_all_plots(file_names_dict, plot_nr_of_spectrum_pairs):
    labels = list(file_names_dict.keys())
    directory = "../../../../data/general_benchmarking"
    folder_name = ""
    for label in labels:
        folder_name += label + "_"
    folder_name += "plots"

    folder_path = os.path.join(directory, folder_name)
    #Make plots folder if it does not exist
    if not os.path.exists(folder_path):
        assert not os.path.isfile(folder_path), "The folder specified is a file"
        os.mkdir(folder_path)
    for validation_type in ("negative_negative", "negative_positive",
                            "positive_positive", "both_both"):
        plot_multiple_rms_per_bin(files_to_plot, validation_type, folder_path, plot_nr_of_spectrum_pairs)


if __name__ == "__main__":
    # model_folder_name = "ms2deepscore_model_both_mode_precursor_mz_ionmode_500_500_layers"
    # data_dir = "../../../../data/"
    # model_folder = f"../../../../data/trained_models/{model_folder_name}"
    # files_to_plot = {"without_instrument_type": "ms2deepscore_model_both_mode_precursor_mz_ionmode_500_500_layers",
    #                  "with_instrument_type":  "ms2deepscore_model_both_mode_precursor_mz_ionmode_source_instrument_500_500_layers"}
    files_to_plot = {"both": "inchikey_generator/both_mode_precursor_mz_ionmode_500_500_layers",
                     "positive": "inchikey_generator/positive_mode_precursor_mz_500_500_layers",
                     "negative": "inchikey_generator/negative_mode_precursor_mz_500_500_layers"}

    # files_to_plot = {"200_embeddings_s": "both_mode_precursor_mz_ionmode_500_500_layers_200_embedding",
    #                  "800_embeddings_s": "both_mode_precursor_mz_ionmode_2000_2000_layers_800_embedding"}
    # files_to_plot = {"inchikey_generator": "inchikey_generator/both_mode_precursor_mz_ionmode_500_500_layers",
    #                  "spectrum_generator_both_p_i": "spectrum_generator/both_mode_precursor_mz_ionmode_500_500_layers_200_embedding"}

    plot_all_plots(files_to_plot, plot_nr_of_spectrum_pairs=True)
