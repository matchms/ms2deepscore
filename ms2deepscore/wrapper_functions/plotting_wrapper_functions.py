import os
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.benchmarking.plot_average_per_bin import plot_average_per_bin
from ms2deepscore.benchmarking.plot_heatmaps import create_3_heatmaps
from ms2deepscore.benchmarking.plot_rmse_per_bin import plot_loss_per_bin_multiple_benchmarks


def create_plots_between_ionmodes(positive_validation_spectra,
                                  negative_validation_spectra,
                                  model_file_name: str,
                                  results_folder: str,
                                  nr_of_bins=50):
    """
    Creates comparison plots between different ion modes.

    Parameters
    ----------
    positive_validation_spectra:
        The validation spectra in positive ionisation mode
    negative_validation_spectra:
        The validation spectra in negative ionistation mode
    model_file_name:
        MS2Deepscore model file name
    results_folder:
        Directory where the plots will be saved.
    nr_of_bins:
        Nr of tanimoto score bins to use for plotting.
    """
    os.makedirs(results_folder, exist_ok=True)

    scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(model_file_name, positive_validation_spectra,
                                                                    negative_validation_spectra)

    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          nr_of_bins=nr_of_bins,
                                          loss_type="RMSE")
    plt.savefig(os.path.join(results_folder, "RMSE_per_bin.svg"))

    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          nr_of_bins=nr_of_bins,
                                          loss_type="MSE")
    plt.savefig(os.path.join(results_folder, "MSE_per_bin.svg"))

    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          nr_of_bins=nr_of_bins,
                                          loss_type="MAE")
    plt.savefig(os.path.join(results_folder, "MAE_per_bin.svg"))

    fig = create_3_heatmaps(scores_between_all_ionmodes, nr_of_bins)
    fig.savefig(os.path.join(results_folder, "heatmaps.svg"))

    fig = plot_average_per_bin(scores_between_all_ionmodes, nr_of_bins)
    fig.savefig(os.path.join(results_folder, "average_per_bin.svg"))


def create_plots_for_all_models(models_directory,
                                results_folder=None,
                                nr_of_bins=10):
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
                create_plots_between_ionmodes(os.path.join(trained_models_dir, model_dir_name),
                                              results_folder_per_model, nr_of_bins=nr_of_bins)


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
