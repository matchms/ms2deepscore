import os
from typing import List
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from matplotlib import pyplot as plt
from rdkit import Chem
from tqdm import tqdm

from ms2deepscore import MS2DeepScore
from ms2deepscore.train_new_model.spectrum_pair_selection import select_inchi_for_unique_inchikeys
from ms2deepscore.utils import load_pickled_file, save_pickled_file
from ms2deepscore.visualize_results.calculate_tanimoto_scores import get_tanimoto_score_between_spectra
from ms2deepscore.visualize_results.plotting import plot_histograms


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
        true_values = get_tanimoto_score_between_spectra(val_spectra_1, val_spectra_2)
        save_pickled_file(true_values, true_values_file_name)

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
