"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""

import os
from matchms.exporting import save_spectra
from matchms.importing import load_spectra
from ms2deepscore.benchmarking_models.calculate_scores_for_validation import \
    calculate_true_values_and_predictions_for_validation_spectra
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.split_positive_and_negative_mode import \
    split_by_ionmode
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from ms2deepscore.train_new_model.validation_and_test_split import \
    split_spectra_in_random_inchikey_sets
from ms2deepscore.utils import load_spectra_as_list
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import \
    create_plots_between_all_ionmodes


def train_ms2deepscore_wrapper(spectra_file_path,
                               settings: SettingsMS2Deepscore,
                               validation_split_fraction=20
                               ):
    """Splits data, trains a ms2deepscore model, and does benchmarking.

    If the data split was already done, the data split will be reused.

    spectra_file_path:
        The path to the spectra that should be used for training. (it will be split in train, val and test)
    settings:
        An object with the MS2Deepscore settings.
    validation_split_fraction:
        The fraction of the inchikeys that will be used for validation and test.
    """

    stored_training_data = StoreTrainingData(spectra_file_path,
                                             split_fraction=validation_split_fraction)

    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra = stored_training_data.load_training_data(settings.ionisation_mode, "training")
    validation_spectra = stored_training_data.load_training_data(settings.ionisation_mode, "validation")

    # Train model
    train_ms2ds_model(training_spectra, validation_spectra,
                      stored_training_data.trained_models_folder,
                      settings)

    # Create performance plots for validation spectra
    ms2deepsore_model_file_name = os.path.join(stored_training_data.trained_models_folder,
                                               settings.model_directory_name,
                                               settings.model_file_name)
    calculate_true_values_and_predictions_for_validation_spectra(
        positive_validation_spectra=stored_training_data.load_positive_train_split("validation"),
        negative_validation_spectra=stored_training_data.load_negative_train_split("validation"),
        ms2deepsore_model_file_name=ms2deepsore_model_file_name,
        results_directory=os.path.join(stored_training_data.trained_models_folder,
                                       settings.model_directory_name, "benchmarking_results"))

    create_plots_between_all_ionmodes(model_directory=os.path.join(stored_training_data.trained_models_folder,
                                                                   settings.model_directory_name))
    return settings.model_directory_name


class StoreTrainingData:
    """Stores, loads and creates all the training data for a spectrum file.

    This includes splitting positive and negative mode spectra and splitting train, test, val spectra.
    It allows for reusing previously created training data for the creation of additional models.
    To do this, just specify the same spectrum file name and directory."""

    def __init__(self, spectra_file_name,
                 split_fraction=20):
        self.root_directory = os.path.dirname(spectra_file_name)
        assert os.path.isdir(self.root_directory)
        self.spectra_file_name = spectra_file_name
        assert os.path.isfile(self.spectra_file_name)
        self.split_fraction = split_fraction
        self.trained_models_folder = os.path.join(self.root_directory, "trained_models")
        os.makedirs(self.trained_models_folder, exist_ok=True)

        self.training_and_val_dir = os.path.join(self.root_directory, "training_and_validation_split")
        os.makedirs(self.training_and_val_dir, exist_ok=True)

        self.positive_negative_split_dir = os.path.join(self.root_directory, "pos_neg_split")
        os.makedirs(self.positive_negative_split_dir, exist_ok=True)

        # Spectrum file names
        self.positive_mode_spectra_file = os.path.join(self.positive_negative_split_dir, "positive_spectra.mgf")
        self.negative_mode_spectra_file = os.path.join(self.positive_negative_split_dir, "negative_spectra.mgf")
        self.positive_validation_spectra_file = os.path.join(self.training_and_val_dir, "positive_validation_spectra.mgf")
        self.positive_training_spectra_file = os.path.join(self.training_and_val_dir, "positive_training_spectra.mgf")
        self.positive_testing_spectra_file = os.path.join(self.training_and_val_dir, "positive_testing_spectra.mgf")
        self.negative_validation_spectra_file = os.path.join(self.training_and_val_dir, "negative_validation_spectra.mgf")
        self.negative_training_spectra_file = os.path.join(self.training_and_val_dir, "negative_training_spectra.mgf")
        self.negative_testing_spectra_file = os.path.join(self.training_and_val_dir, "negative_testing_spectra.mgf")

    def load_positive_mode_spectra(self):
        if os.path.isfile(self.positive_mode_spectra_file):
            return load_spectra_as_list(self.positive_mode_spectra_file)
        positive_mode_spectra, _ = self.split_and_save_positive_and_negative_spectra()
        print("Loaded previously stored positive mode spectra")
        return positive_mode_spectra

    def load_negative_mode_spectra(self):
        if os.path.isfile(self.negative_mode_spectra_file):
            return load_spectra_as_list(self.negative_mode_spectra_file)
        _, negative_mode_spectra = self.split_and_save_positive_and_negative_spectra()
        print("Loaded previously stored negative mode spectra")
        return negative_mode_spectra

    def split_and_save_positive_and_negative_spectra(self):
        assert not os.path.isfile(self.positive_mode_spectra_file), "the positive mode spectra file already exists"
        assert not os.path.isfile(self.negative_mode_spectra_file), "the negative mode spectra file already exists"
        positive_mode_spectra, negative_mode_spectra = split_by_ionmode(
            load_spectra(self.spectra_file_name, metadata_harmonization=False))
        save_spectra(positive_mode_spectra, self.positive_mode_spectra_file)
        save_spectra(negative_mode_spectra, self.negative_mode_spectra_file)
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
                split_spectra_in_random_inchikey_sets(self.load_positive_mode_spectra(), self.split_fraction)
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
                split_spectra_in_random_inchikey_sets(self.load_negative_mode_spectra(), self.split_fraction)
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