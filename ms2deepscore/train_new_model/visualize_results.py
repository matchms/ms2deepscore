import os
import numpy as np
from matplotlib import pyplot as plt
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from ms2deepscore.plotting import create_histograms_plot
from ms2deepscore.utils import load_pickled_file, save_pickled_file


def get_tanimoto_indexes(tanimoto_df, spectra):
    inchikey_idx_reference_spectra_1 = np.zeros(len(spectra))
    for i, spec in enumerate(spectra):
        inchikey_idx_reference_spectra_1[i] = np.where(
            tanimoto_df.index.values == spec.get("inchikey")[:14]
        )[0]
    return inchikey_idx_reference_spectra_1.astype("int")


def get_correct_predictions(tanimoto_df, reference_spectra_1, reference_spectra_2):
    inchikey_idx_1 = get_tanimoto_indexes(tanimoto_df, reference_spectra_1)
    inchikey_idx_2 = get_tanimoto_indexes(tanimoto_df, reference_spectra_2)

    scores_ref = tanimoto_df.values[np.ix_(inchikey_idx_1[:], inchikey_idx_2[:])].copy()
    return scores_ref


def create_plot(predictions, scores_ref, plots_folder, file_name):
    create_histograms_plot(
        scores_ref,
        predictions,
        n_bins=10,
        hist_resolution=100,
        ref_score_name="Tanimoto similarity",
        compare_score_name="MS2DeepScore (predicted Tanimoto similarity)",
    )

    # Make plots folder if it does not exist
    if not os.path.exists(plots_folder):
        assert not os.path.isfile(plots_folder), "The folder specified is a file"
        os.mkdir(plots_folder)

    filename = os.path.join(plots_folder, f"{file_name}_plot.svg")
    plt.savefig(filename)


def create_reverse_plot(predictions, scores_ref, plots_folder, file_name):
    create_histograms_plot(
        predictions,
        scores_ref,
        n_bins=10,
        hist_resolution=100,
        ref_score_name="MS2DeepScore (predicted Tanimoto similarity)",
        compare_score_name="Tanimoto similarity",
    )

    # Make plots folder if it does not exist
    if not os.path.exists(plots_folder):
        assert not os.path.isfile(plots_folder), "The folder specified is a file"
        os.mkdir(plots_folder)

    filename = os.path.join(plots_folder, f"{file_name}_plot_reversed.svg")
    plt.savefig(filename)


def benchmark_wrapper(
    val_spectra_1,
    val_spectra_2,
    benchmarking_results_folder,
    tanimoto_df,
    ms2ds_model,
    file_name,
):
    # pylint: disable=too-many-arguments, too-many-locals
    is_symmetric = (val_spectra_1 == val_spectra_2)

    # Create predictions
    predictions_file_name = os.path.join(
        benchmarking_results_folder, f"{file_name}_predictions.pickle"
    )
    if os.path.exists(predictions_file_name):
        predictions = load_pickled_file(predictions_file_name)
        print(f"Loaded in previous predictions for: {file_name}")
    else:
        predictions = ms2ds_model.matrix(
            val_spectra_1, val_spectra_2, is_symmetric=is_symmetric
        )
        save_pickled_file(predictions, predictions_file_name)

    # Calculate true values
    true_values_file_name = os.path.join(
        benchmarking_results_folder, f"{file_name}_true_values.pickle"
    )
    if os.path.exists(true_values_file_name):
        true_values = load_pickled_file(true_values_file_name)
        print(f"Loaded in previous true values for: {file_name}")
    else:
        true_values = get_correct_predictions(tanimoto_df, val_spectra_1, val_spectra_2)
        save_pickled_file(true_values, true_values_file_name)

    # Create plots
    create_plot(
        predictions,
        true_values,
        plots_folder=os.path.join(benchmarking_results_folder, "plots_normalized_auc"),
        file_name=file_name,
    )
    create_reverse_plot(
        predictions,
        true_values,
        plots_folder=os.path.join(benchmarking_results_folder, "plots_normalized_auc"),
        file_name=file_name,
    )

    mae = np.abs(predictions - true_values).mean()
    rmse = np.sqrt(np.square(predictions - true_values).mean())
    summary = f"For {file_name} the mae={mae} and rmse={rmse}\n"

    averages_summary_file = os.path.join(
        benchmarking_results_folder, "RMSE_and_MAE.txt"
    )
    with open(averages_summary_file, "a", encoding="uft-8") as f:
        f.write(summary)
    print(summary)


def create_all_plots(model_folder_name):
    data_dir = "../../../../data/"
    model_folder = f"../../../../data/trained_models/{model_folder_name}"

    positive_validation_spectra = load_pickled_file(
        os.path.join(
            data_dir,
            "training_and_validation_split",
            "positive_validation_spectra.pickle",
        )
    )
    negative_validation_spectra = load_pickled_file(
        os.path.join(
            data_dir,
            "training_and_validation_split",
            "negative_validation_spectra.pickle",
        )
    )

    # Check if the model already finished training
    if not os.path.exists(os.path.join(model_folder, "history.txt")):
        print(f"Did not plot since {model_folder_name} did not yet finish training")
        return None

    # Create benchmarking results folder
    benchmarking_results_folder = os.path.join(model_folder, "benchmarking_results")
    if not os.path.exists(benchmarking_results_folder):
        assert not os.path.isfile(
            benchmarking_results_folder
        ), "The folder specified is a file"
        os.mkdir(benchmarking_results_folder)

    # Load in MS2Deepscore model
    ms2deepscore_model = load_model(
        os.path.join(model_folder, "ms2deepscore_model.hdf5")
    )
    similarity_score = MS2DeepScore(ms2deepscore_model)

    tanimoto_df = load_pickled_file(
        os.path.join(data_dir, "tanimoto_scores/validation_tanimoto_scores.pickle")
    )

    benchmark_wrapper(
        positive_validation_spectra,
        positive_validation_spectra,
        benchmarking_results_folder,
        tanimoto_df,
        similarity_score,
        "positive_positive",
    )
    benchmark_wrapper(
        negative_validation_spectra,
        negative_validation_spectra,
        benchmarking_results_folder,
        tanimoto_df,
        similarity_score,
        "negative_negative",
    )
    benchmark_wrapper(
        negative_validation_spectra,
        positive_validation_spectra,
        benchmarking_results_folder,
        tanimoto_df,
        similarity_score,
        "negative_positive",
    )
    benchmark_wrapper(
        positive_validation_spectra + negative_validation_spectra,
        positive_validation_spectra + negative_validation_spectra,
        benchmarking_results_folder,
        tanimoto_df,
        similarity_score,
        "both_both",
    )
    return None
