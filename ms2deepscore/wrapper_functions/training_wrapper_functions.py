"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""

import itertools
import os
import pickle
from datetime import datetime
from tqdm import tqdm
from matchms.exporting import save_spectra
from matchms.importing import load_spectra

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      train)
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model import TrainingBatchGenerator, create_spectrum_pair_generator
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import ValidationLossCalculator
from ms2deepscore.train_new_model.train_ms2deepscore import \
    train_ms2ds_model, plot_history, save_history
from ms2deepscore.train_new_model.validation_and_test_split import \
    split_spectra_in_random_inchikey_sets
from ms2deepscore.utils import load_spectra_as_list
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import \
    create_plots_between_ionmodes


def train_ms2deepscore_wrapper(settings: SettingsMS2Deepscore,
                               ):
    """Splits data, trains a ms2deepscore model, and does benchmarking.

    If the data split was already done, the data split will be reused.

    spectra_file_path:
        The path to the spectra that should be used for training. (it will be split in train, val and test)
    settings:
        An object with the MS2Deepscore model settings.
    validation_split_fraction:
        The fraction of the inchikeys that will be used for validation and test.
    """
    split_data_if_necessary(settings)

    training_spectra = load_spectra_in_ionmode(settings.training_spectra_file_name, settings.ionisation_mode)
    validation_spectra = load_spectra_in_ionmode(settings.validation_spectra_file_name, settings.ionisation_mode)

    # Train model
    _, history = train_ms2ds_model(training_spectra, validation_spectra, settings.model_directory_name, settings)

    ms2ds_history_plot_file_name = os.path.join(settings.model_directory_name, settings.history_plot_file_name)
    plot_history(history["losses"], history["val_losses"], ms2ds_history_plot_file_name)
    save_history(os.path.join(settings.model_directory_name, "history.json"), history)

    scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(
        os.path.join(settings.model_directory_name, settings.model_file_name),
        load_spectra_in_ionmode(settings.validation_spectra_file_name, "positive"),
        load_spectra_in_ionmode(settings.validation_spectra_file_name, "negative"),
        settings.fingerprint_type, settings.fingerprint_nbits)

    # Create performance plots for validation spectra
    create_plots_between_ionmodes(scores_between_all_ionmodes,
                                  results_folder=os.path.join(settings.model_directory_name, "benchmarking_results"),
                                  nr_of_bins=50)
    return settings.model_directory_name


def parameter_search(
        base_settings: SettingsMS2Deepscore,
        setting_variations,
        loss_types=("mse",),
        path_checkpoint="results_checkpoint.pkl"
):
    """Runs a grid search.

    If the data split was already done, the data split will be reused.

    spectra_file_path_or_dir:
        The path to the spectra that should be used for training. (it will be split in train, val and test).
        Or the path in which an already existing split is present in a subfolder train_and_validation_split.
    base_settings:
        An object with the MS2Deepscore model settings.
    validation_split_fraction:
        The fraction of the inchikeys that will be used for validation and test.
    """
    print("Initialize Stored Data")
    split_data_if_necessary(base_settings)
    validation_spectra = load_spectra_in_ionmode(base_settings.validation_spectra_file_name, base_settings.ionisation_mode)

    print("Load training data")
    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra = load_spectra_in_ionmode(base_settings.training_spectra_file_name, base_settings.ionisation_mode)

    print("Load validation data")
    validation_spectra = load_spectra_in_ionmode(base_settings.validation_spectra_file_name, base_settings.ionisation_mode)

    # For ionmode sensitive evaluations
    positive_validation_spectra = load_spectra_in_ionmode(base_settings.validation_spectra_file_name, "positive")
    negative_validation_spectra = load_spectra_in_ionmode(base_settings.validation_spectra_file_name, "negative")

    results = {}
    train_generator = None

    # Generate all combinations of setting variations
    keys, values = zip(*setting_variations.items())
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        settings_dict = base_settings.get_dict()
        settings_dict.update(params)
        settings = SettingsMS2Deepscore(**settings_dict)
        settings.time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


        print(f"Testing combination: {params}")
        # TODO (mabye): implement smarter way to now always re-initialize the generators
        #  fields_affecting_generators = [
        #     "fingerprint_type",
        #     "fingerprint_nbits",
        #     "max_pairs_per_bin",
        #     "same_prob_bins",
        #     "include_diagonal",
        #     "mz_bin_width",
        #  ]
        #  search_includes_generator_parameters = False
        #  for field in fields_affecting_generators:
        #     if field in keys:
        #         search_includes_generator_parameters = True
        #  if search_includes_generator_parameters or (train_generator is None):

        # Make folder and save settings
        os.makedirs(settings.model_directory_name, exist_ok=True)
        settings.save_to_file(os.path.join(settings.model_directory_name, "settings.json"))
        # Create a training generator
        spectrum_pair_generator = create_spectrum_pair_generator(training_spectra, settings=settings)
        train_generator = TrainingBatchGenerator(spectrum_pair_generator=spectrum_pair_generator, settings=settings)
        # Create a validation loss calculator
        validation_loss_calculator = ValidationLossCalculator(validation_spectra,
                                                              settings=settings)

        model = SiameseSpectralModel(settings=settings)

        output_model_file_name = os.path.join(settings.model_directory_name, settings.model_file_name)

        # Train model
        try:
            history = train(
                model,
                train_generator,
                num_epochs=settings.epochs,
                learning_rate=settings.learning_rate,
                validation_loss_calculator=validation_loss_calculator,
                patience=settings.patience,
                loss_function=settings.loss_function,
                checkpoint_filename=output_model_file_name,
                lambda_l1=0, lambda_l2=0
            )
        except Exception as error:
            print("An exception occurred:", error)
            print("---- Model training failed! ----")
            print("---- Settings ----")
            print(settings.get_dict())
            continue

        scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(
            model_file_name=os.path.join(settings.model_directory_name, settings.model_file_name),
            positive_validation_spectra=positive_validation_spectra,
            negative_validation_spectra=negative_validation_spectra)

        combination_results = {
            "params": params,
            "history": history,
            "losses": {}
        }
        for loss_type in loss_types:
            losses_per_ionmode = {}
            for predictions_and_tanimoto_scores in scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores():
                _, losses = predictions_and_tanimoto_scores.get_average_loss_per_bin_per_inchikey_pair(loss_type,
                                                                                                       settings.same_prob_bins)
                losses_per_ionmode[predictions_and_tanimoto_scores.label] = losses
            combination_results["losses"][loss_type] = losses_per_ionmode

        # Store results
        combination_key = tuple(params.items())
        results[combination_key] = combination_results

        # Store checkpoint
        with open(path_checkpoint, 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    return results


def split_data_if_necessary(settings: SettingsMS2Deepscore):
    """Splits and saves spectra into train, val and test"""
    if settings.spectrum_file_path is None:
        raise ValueError("Please specify a spectrum_file_path in SettingsMS2DeepScore")
    os.makedirs(settings.results_folder, exist_ok=True)
    nr_of_files_existing = sum(os.path.exists(f) for f in (settings.validation_spectra_file_name,
                                                     settings.test_spectra_file_name,
                                                     settings.training_spectra_file_name))
    if nr_of_files_existing == 3:
        # all files already exist
        print("Using existing spectrum split")
    elif nr_of_files_existing == 0:
        # The files don't exist yet
        print("Splitting the data into train, test and validation")
        spectra = load_spectra_as_list(settings.spectrum_file_path)
        validation_spectra, test_spectra, train_spectra = split_spectra_in_random_inchikey_sets(
            spectra, k=settings.train_test_split_fraction, random_seed=settings.random_seed)

        save_spectra(validation_spectra, settings.validation_spectra_file_name)
        save_spectra(test_spectra, settings.test_spectra_file_name)
        save_spectra(train_spectra, settings.training_spectra_file_name)
    else:
        raise ValueError("Some of the validation files do exist and some don't")

def load_spectra_in_ionmode(spectrum_file_name, ionmode: str):
    selected_spectra = []
    if ionmode == "both":
        return list(tqdm(load_spectra(spectrum_file_name), desc=f"loading spectra from {spectrum_file_name} in both ionmodes"))
    else:
        if ionmode not in ("positive", "negative"):
            raise ValueError("Expected ionmode to be 'positive' or 'negative'")
        for spectrum in tqdm(load_spectra(spectrum_file_name), desc=f"Loading spectra in {ionmode} ionmode"):
            if spectrum.get("ionmode") == ionmode:
                selected_spectra.append(spectrum)
        return selected_spectra