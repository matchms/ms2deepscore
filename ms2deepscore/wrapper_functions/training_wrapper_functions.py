"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""

import itertools
import os
import pickle
from typing import Optional
from datetime import datetime
from matchms import Spectrum
from matchms.exporting import save_spectra
from matchms.importing import load_spectra

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      train)
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import ValidationLossCalculator
from ms2deepscore.train_new_model.data_generators import create_data_generator
from ms2deepscore.train_new_model.split_positive_and_negative_mode import \
    split_by_ionmode
from ms2deepscore.train_new_model.train_ms2deepscore import \
    train_ms2ds_model, plot_history, save_history
from ms2deepscore.train_new_model.validation_and_test_split import \
    split_spectra_in_random_inchikey_sets
from ms2deepscore.utils import load_spectra_as_list
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import \
    create_plots_between_ionmodes


def train_ms2deepscore_wrapper(spectra_file_path,
                               settings: SettingsMS2Deepscore,
                               validation_split_fraction=20
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

    stored_training_data = StoreTrainingData(spectra_file_path,
                                             split_fraction=validation_split_fraction,
                                             random_seed=settings.random_seed)

    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra = stored_training_data.load_training_data(settings.ionisation_mode, "training")
    validation_spectra = stored_training_data.load_training_data(settings.ionisation_mode, "validation")

    model_directory_name = create_model_directory_name(settings)
    results_folder = os.path.join(stored_training_data.trained_models_folder, model_directory_name)

    # Train model
    _, history = train_ms2ds_model(
        training_spectra, validation_spectra,
        results_folder,
        settings
    )

    ms2ds_history_plot_file_name = os.path.join(results_folder, settings.history_plot_file_name)
    plot_history(history["losses"], history["val_losses"], ms2ds_history_plot_file_name)
    save_history(os.path.join(results_folder, "history.json"), history)

    scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(
        os.path.join(stored_training_data.trained_models_folder, model_directory_name, settings.model_file_name),
        stored_training_data.load_training_data("positive", "validation"),
        stored_training_data.load_training_data("negative", "validation"),
        settings.fingerprint_type, settings.fingerprint_nbits)

    # Create performance plots for validation spectra
    create_plots_between_ionmodes(scores_between_all_ionmodes,
                                  results_folder=os.path.join(stored_training_data.trained_models_folder,
                                                              model_directory_name, "benchmarking_results"),
                                  nr_of_bins=50)
    return model_directory_name


def parameter_search(
        spectra_file_path_or_dir: str,
        base_settings: SettingsMS2Deepscore,
        setting_variations,
        validation_split_fraction=20,
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
    stored_training_data = StoreTrainingData(spectra_file_path_or_dir,
                                             split_fraction=validation_split_fraction,
                                             random_seed=base_settings.random_seed)

    print("Load training data")
    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra = stored_training_data.load_training_data(base_settings.ionisation_mode, "training")

    print("Load validation data")
    validation_spectra = stored_training_data.load_training_data(base_settings.ionisation_mode, "validation")

    # For ionmode sensitive evaluations
    positive_validation_spectra = stored_training_data.load_positive_train_split("validation")
    negative_validation_spectra = stored_training_data.load_negative_train_split("validation")

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

        # Set folder name for storing model and training progress
        model_directory_name = create_model_directory_name(settings)
        results_folder = os.path.join(stored_training_data.trained_models_folder, model_directory_name)

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
        os.makedirs(results_folder, exist_ok=True)
        settings.save_to_file(os.path.join(results_folder, "settings.json"))
        # Create a training generator
        train_generator = create_data_generator(training_spectra, settings)
        # Create a validation loss calculator
        validation_loss_calculator = ValidationLossCalculator(validation_spectra,
                                                              settings=settings)

        model = SiameseSpectralModel(settings=settings)

        output_model_file_name = os.path.join(results_folder, settings.model_file_name)

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
            model_file_name=os.path.join(stored_training_data.trained_models_folder,
                                         model_directory_name,
                                         settings.model_file_name),
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


def create_model_directory_name(settings: SettingsMS2Deepscore):
    """Creates a directory name using metadata, it will contain the metadata, the binned spectra and final model"""
    binning_file_label = ""
    for metadata_generator in settings.additional_metadata:
        binning_file_label += metadata_generator[1]["metadata_field"] + "_"

    # Define a neural net structure label
    neural_net_structure_label = ""
    for layer in settings.base_dims:
        neural_net_structure_label += str(layer) + "_"
    neural_net_structure_label += "layers"

    if settings.embedding_dim:
        neural_net_structure_label += f"_{str(settings.embedding_dim)}_embedding"
    model_folder_file_name = f"{settings.ionisation_mode}_mode_{binning_file_label}" \
                             f"{neural_net_structure_label}_{settings.time_stamp}"
    print(f"The model will be stored in the folder: {model_folder_file_name}")
    return model_folder_file_name


class StoreTrainingData:
    """Stores, loads, and creates all the training data for a spectrum file or existing split.

    This includes splitting positive and negative mode spectra and splitting train, test, and val spectra.
    It allows for reusing previously created training data for the creation of additional models.
    Users can either provide a single spectra file or an existing folder with split data.
    """

    def __init__(self,
                 spectra_file_path_or_dir: str,
                 split_fraction: int = 20,
                 random_seed: Optional[int] = None):
        self.root_directory = spectra_file_path_or_dir
        self.split_fraction = split_fraction
        self.random_seed = random_seed

        # Check if the input is a directory with pre-split data or a spectra file
        if os.path.isdir(spectra_file_path_or_dir):
            self.spectra_file_name = None  # No specific spectra file
            self.root_directory = spectra_file_path_or_dir
            print("Starting from an existing split directory.")
        elif os.path.isfile(spectra_file_path_or_dir):
            self.spectra_file_name = spectra_file_path_or_dir  # Input is a spectra file
            self.root_directory = os.path.dirname(spectra_file_path_or_dir)
            print(f"Starting from spectra file: {self.spectra_file_name}")
        else:
            raise ValueError("The provided path is neither a valid directory nor a spectra file.")

        # Define folders for training and validation splits
        self.trained_models_folder = os.path.join(self.root_directory, "trained_models")
        os.makedirs(self.trained_models_folder, exist_ok=True)

        self.training_and_val_dir = os.path.join(self.root_directory, "training_and_validation_split")
        os.makedirs(self.training_and_val_dir, exist_ok=True)

        self.positive_negative_split_dir = os.path.join(self.root_directory, "pos_neg_split")
        os.makedirs(self.positive_negative_split_dir, exist_ok=True)

        # Spectrum file paths for splits
        self.positive_mode_spectra_file = os.path.join(self.positive_negative_split_dir, "positive_spectra.mgf")
        self.negative_mode_spectra_file = os.path.join(self.positive_negative_split_dir, "negative_spectra.mgf")
        self.positive_validation_spectra_file = os.path.join(self.training_and_val_dir,
                                                             "positive_validation_spectra.mgf")
        self.positive_training_spectra_file = os.path.join(self.training_and_val_dir, "positive_training_spectra.mgf")
        self.positive_testing_spectra_file = os.path.join(self.training_and_val_dir, "positive_testing_spectra.mgf")
        self.negative_validation_spectra_file = os.path.join(self.training_and_val_dir,
                                                             "negative_validation_spectra.mgf")
        self.negative_training_spectra_file = os.path.join(self.training_and_val_dir, "negative_training_spectra.mgf")
        self.negative_testing_spectra_file = os.path.join(self.training_and_val_dir, "negative_testing_spectra.mgf")

    def load_positive_mode_spectra(self) -> list[Spectrum]:
        """Load or split positive mode spectra."""
        if os.path.isfile(self.positive_mode_spectra_file):
            print("Loading previously stored positive mode spectra.")
            return load_spectra_as_list(self.positive_mode_spectra_file)

        if not self.spectra_file_name:
            raise ValueError("No spectra file provided and no pre-split data available for positive mode spectra.")

        positive_mode_spectra, _ = self.split_and_save_positive_and_negative_spectra()
        print("Loaded positive mode spectra")
        return positive_mode_spectra

    def load_negative_mode_spectra(self) -> list[Spectrum]:
        """Load or split negative mode spectra."""
        if os.path.isfile(self.negative_mode_spectra_file):
            print("Loading previously stored negative mode spectra.")
            return load_spectra_as_list(self.negative_mode_spectra_file)

        if not self.spectra_file_name:
            raise ValueError("No spectra file provided and no pre-split data available for negative mode spectra.")

        _, negative_mode_spectra = self.split_and_save_positive_and_negative_spectra()
        print("Loaded negative mode spectra")
        return negative_mode_spectra

    def split_and_save_positive_and_negative_spectra(self):
        """Split the spectra into positive and negative modes and save them."""
        if not self.spectra_file_name:
            raise ValueError("No spectra file provided for splitting")
        if os.path.isfile(self.positive_mode_spectra_file):
            raise ValueError("The positive mode spectra file already exists")
        if os.path.isfile(self.negative_mode_spectra_file):
            raise ValueError("The negative mode spectra file already exists")
        positive_mode_spectra, negative_mode_spectra = split_by_ionmode(
            load_spectra(self.spectra_file_name, metadata_harmonization=True))
        save_spectra(positive_mode_spectra, self.positive_mode_spectra_file)
        save_spectra(negative_mode_spectra, self.negative_mode_spectra_file)
        print("Separated and stored positive mode and negative mode spectra.")
        return positive_mode_spectra, negative_mode_spectra

    def load_positive_train_split(self, spectra_type):
        if spectra_type == "training":
            spectra_file_name = self.positive_training_spectra_file
        elif spectra_type == "validation":
            spectra_file_name = self.positive_validation_spectra_file
        elif spectra_type == "testing":
            spectra_file_name = self.positive_testing_spectra_file
        else:
            raise ValueError("Expected 'training', 'validation' or 'testing' as spectra_type")

        # If it could not be loaded do the data split and save the files.
        if not os.path.isfile(spectra_file_name):
            positive_validation_spectra, positive_testing_spectra, positive_training_spectra = \
                split_spectra_in_random_inchikey_sets(self.load_positive_mode_spectra(),
                                                      self.split_fraction, self.random_seed)
            save_spectra(positive_training_spectra, self.positive_training_spectra_file)
            save_spectra(positive_validation_spectra, self.positive_validation_spectra_file)
            save_spectra(positive_testing_spectra, self.positive_testing_spectra_file)
            print(f"Positive split \n Train: {len(positive_training_spectra)} \n "
                  f"Validation: {len(positive_validation_spectra)} \n Test: {len(positive_testing_spectra)}")
        return load_spectra_as_list(spectra_file_name)

    def load_negative_train_split(self,
                                  spectra_type: str):
        if spectra_type == "training":
            spectra_file_name = self.negative_training_spectra_file
        elif spectra_type == "validation":
            spectra_file_name = self.negative_validation_spectra_file
        elif spectra_type == "testing":
            spectra_file_name = self.negative_testing_spectra_file
        else:
            raise ValueError("Expected 'training', 'validation' or 'testing' as spectra_type")

        # If it could not be loaded do the data split and save the files.
        if not os.path.isfile(spectra_file_name):
            negative_validation_spectra, negative_testing_spectra, negative_training_spectra = \
                split_spectra_in_random_inchikey_sets(self.load_negative_mode_spectra(),
                                                      self.split_fraction, self.random_seed)
            save_spectra(negative_training_spectra, self.negative_training_spectra_file)
            save_spectra(negative_validation_spectra, self.negative_validation_spectra_file)
            save_spectra(negative_testing_spectra, self.negative_testing_spectra_file)
            print(f"negative split \n Train: {len(negative_training_spectra)} \n "
                  f"Validation: {len(negative_validation_spectra)} \n Test: {len(negative_testing_spectra)}")
        return load_spectra_as_list(spectra_file_name)

    def load_training_data(self,
                           ionisation_mode: str,
                           data_split_type: str):
        """Loads the spectra for a specified mode."""
        if ionisation_mode == "positive":
            return self.load_positive_train_split(data_split_type)
        if ionisation_mode == "negative":
            return self.load_negative_train_split(data_split_type)
        if ionisation_mode == "both":
            return self.load_positive_train_split(data_split_type) + self.load_negative_train_split(data_split_type)
        raise ValueError("expected ionisation mode to be 'positive', 'negative' or 'both'")
