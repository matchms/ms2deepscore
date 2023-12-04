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


def get_tanimoto_score_between_spectra(spectra_1: List[Spectrum],
                                       spectra_2: List[Spectrum]):
    """Gets the tanimoto scores between two list of spectra

    It is optimized by calculating the tanimoto scores only between unique fingerprints/smiles.
    The tanimoto scores are derived after.

    """
    def get_tanimoto_indexes(tanimoto_df, spectra):
        inchikey_idx_reference_spectra_1 = np.zeros(len(spectra))
        for i, spec in enumerate(spectra):
            inchikey_idx_reference_spectra_1[i] = np.where(tanimoto_df.index.values == spec.get("inchikey")[:14])[0]
        return inchikey_idx_reference_spectra_1.astype("int")

    tanimoto_df = calculate_tanimoto_scores_unique_inchikey(spectra_1, spectra_2)
    inchikey_idx_1 = get_tanimoto_indexes(tanimoto_df, spectra_1)
    inchikey_idx_2 = get_tanimoto_indexes(tanimoto_df, spectra_2)

    scores_ref = tanimoto_df.values[np.ix_(inchikey_idx_1[:], inchikey_idx_2[:])].copy()
    return scores_ref


def calculate_tanimoto_scores_unique_inchikey(list_of_spectra_1: List[Spectrum],
                                              list_of_spectra_2: List[Spectrum]):
    """Returns a dataframe with the tanimoto scores between each unique inchikey in list of spectra"""

    def get_fingerprint(smiles: str):
        fingerprint = np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), fpSize=2048))
        assert isinstance(fingerprint, np.ndarray), f"Fingerprint could not be set smiles is {smiles}"
        return fingerprint

    spectra_with_most_frequent_inchi_per_inchikey_1, unique_inchikeys_1 = \
        select_inchi_for_unique_inchikeys(list_of_spectra_1)
    spectra_with_most_frequent_inchi_per_inchikey_2, unique_inchikeys_2 = \
        select_inchi_for_unique_inchikeys(list_of_spectra_2)

    list_of_smiles_1 = [spectrum.get("smiles") for spectrum in spectra_with_most_frequent_inchi_per_inchikey_1]
    list_of_smiles_2 = [spectrum.get("smiles") for spectrum in spectra_with_most_frequent_inchi_per_inchikey_2]

    fingerprints_1 = np.array([get_fingerprint(spectrum) for spectrum in tqdm(list_of_smiles_1,
                                                                              desc="Calculating fingerprints")])
    fingerprints_2 = np.array([get_fingerprint(spectrum) for spectrum in tqdm(list_of_smiles_2,
                                                                              desc="Calculating fingerprints")])
    print("Calculating tanimoto scores")
    tanimoto_scores = jaccard_similarity_matrix(fingerprints_1, fingerprints_2)
    tanimoto_df = pd.DataFrame(tanimoto_scores, index=unique_inchikeys_1, columns=unique_inchikeys_2)
    return tanimoto_df
