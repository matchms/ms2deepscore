"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""
import os
from matchms.exporting.save_spectra import save_spectra
from matchms.importing.load_spectra import load_spectra
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.split_positive_and_negative_mode import \
    split_pos_and_neg
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from ms2deepscore.train_new_model.validation_and_test_split import \
    split_spectra_in_random_inchikey_sets


def train_ms2deepscore_wrapper(data_directory,
                               spectra_file_name,
                               settings: SettingsMS2Deepscore,
                               validation_split_fraction = 20
                               ):
    """Trains a ms2deepscore model, including the data split into pos,val,train spectra."""
    directory_structure = StoreTrainingData(data_directory, spectra_file_name, validation_split_fraction)

    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra, validation_spectra, test_spectra = load_train_val_data(directory_structure,
                                                                             settings.ionisation_mode)
    # Train model
    train_ms2ds_model(training_spectra, validation_spectra,
                      os.path.join(directory_structure.trained_models_folder, settings.model_directory_name),
                      settings)
    # todo add creation of benchmarking?
    # todo store the settings as well in the settings.model_directory_name
    return settings.model_directory_name


def load_train_val_data(directory_structure: "StoreTrainingData",
                        ionisation_mode: str):
    """Loads the train, val and test spectra for a specified mode."""
    if ionisation_mode == "positive":
        return directory_structure.load_positive_train_split()
    if ionisation_mode == "negative":
        return directory_structure.load_negative_train_split()
    if ionisation_mode == "both":
        return directory_structure.load_both_mode_train_split()
    raise ValueError("expected ionisation mode to be 'positive', 'negative' or 'both'")


class StoreTrainingData:
    """Stores, loads and creates all the training data for a spectrum file.

    This includes splitting positive and negative mode spectra and splitting train, test, val spectra.
    It allows for reusing previously created training data for the creation of additional models.
    To do this, just specify the same spectrum file name and directory."""

    def __init__(self, root_directory, spectra_file_name,
                 split_fraction=20):
        # todo add the spectrum file name to all created file names
        # todo only use the spectrum file name and not root_directory.
        self.root_directory = root_directory
        assert os.path.isdir(self.root_directory)
        self.spectra_file_name = os.path.join(self.root_directory, spectra_file_name)
        assert os.path.isfile(self.spectra_file_name)
        self.split_fraction = split_fraction
        self.trained_models_folder = os.path.join(self.root_directory, "trained_models")
        os.makedirs(self.trained_models_folder, exist_ok=True)

        self.training_and_val_dir = os.path.join(self.root_directory, "training_and_validation_split")
        os.makedirs(self.training_and_val_dir, exist_ok=True)

        self.positive_negative_split_dir = os.path.join(self.root_directory, "pos_neg_split")
        # Check if the folder exists otherwise make new folder
        os.makedirs(self.positive_negative_split_dir, exist_ok=True)

        self.positive_mode_spectra_file = os.path.join(self.positive_negative_split_dir, "positive_spectra.mgf")
        self.negative_mode_spectra_file = os.path.join(self.positive_negative_split_dir, "negative_spectra.mgf")
        self.positive_validation_spectra_file = os.path.join(self.training_and_val_dir, "positive_validation_spectra.mgf")
        self.positive_training_spectra_file = os.path.join(self.training_and_val_dir, "positive_training_spectra.mgf")
        self.positive_testing_spectra_file = os.path.join(self.training_and_val_dir, "positive_testing_spectra.mgf")
        self.negative_validation_spectra_file = os.path.join(self.training_and_val_dir, "negative_validation_spectra.mgf")
        self.negative_training_spectra_file = os.path.join(self.training_and_val_dir, "negative_training_spectra.mgf")
        self.negative_testing_spectra_file = os.path.join(self.training_and_val_dir, "negative_testing_spectra.mgf")

    def get_all_spectra(self):
        return load_spectra(self.spectra_file_name)

    def load_positive_mode_spectra(self):
        if os.path.isfile(self.positive_mode_spectra_file):
            return load_spectra(self.positive_mode_spectra_file)
        positive_mode_spectra, negative_mode_spectra = self.split_and_save_positive_and_negative_spectra()
        print("Loaded previously stored positive mode spectra")
        return positive_mode_spectra

    def load_negative_mode_spectra(self):
        if os.path.isfile(self.negative_mode_spectra_file):
            return load_spectra(self.negative_mode_spectra_file)
        positive_mode_spectra, negative_mode_spectra = self.split_and_save_positive_and_negative_spectra()
        print("Loaded previously stored negative mode spectra")
        return negative_mode_spectra

    def split_and_save_positive_and_negative_spectra(self):
        assert not os.path.isfile(self.positive_mode_spectra_file), "the positive mode spectra file already exists"
        assert not os.path.isfile(self.negative_mode_spectra_file), "the negative mode spectra file already exists"
        spectra = self.get_all_spectra()
        positive_mode_spectra, negative_mode_spectra = split_pos_and_neg(spectra)
        save_spectra(positive_mode_spectra, self.positive_mode_spectra_file)
        save_spectra(negative_mode_spectra, self.negative_mode_spectra_file)
        return positive_mode_spectra, negative_mode_spectra

    def load_positive_train_split(self):
        all_files_exist = True
        for spectra_file in [self.positive_training_spectra_file,
                             self.positive_testing_spectra_file,
                             self.positive_validation_spectra_file, ]:
            if not os.path.isfile(spectra_file):
                all_files_exist = False

        if all_files_exist:
            positive_training_spectra = load_spectra(self.positive_training_spectra_file)
            positive_validation_spectra = load_spectra(self.positive_validation_spectra_file)
            positive_testing_spectra = load_spectra(self.positive_testing_spectra_file)
        else:
            positive_validation_spectra, positive_testing_spectra, positive_training_spectra = \
                split_spectra_in_random_inchikey_sets(self.load_positive_mode_spectra(), self.split_fraction)
            save_spectra(positive_training_spectra, self.positive_training_spectra_file)
            save_spectra(positive_validation_spectra, self.positive_validation_spectra_file)
            save_spectra(positive_testing_spectra, self.positive_testing_spectra_file)
        print(f"Positive split \n "
              f"Train: {len(positive_training_spectra)} \n "
              f"Validation: {len(positive_validation_spectra)} \n "
              f"Test: {len(positive_testing_spectra)}")
        return positive_training_spectra, positive_validation_spectra, positive_testing_spectra

    def load_negative_train_split(self):
        all_files_exist = True
        for spectra_file in [self.negative_training_spectra_file,
                             self.negative_testing_spectra_file,
                             self.negative_validation_spectra_file, ]:
            if not os.path.isfile(spectra_file):
                all_files_exist = False

        if all_files_exist:
            negative_training_spectra = load_spectra(self.negative_training_spectra_file)
            negative_validation_spectra = load_spectra(self.negative_validation_spectra_file)
            negative_testing_spectra = load_spectra(self.negative_testing_spectra_file)
        else:
            negative_validation_spectra, negative_testing_spectra, negative_training_spectra = \
                split_spectra_in_random_inchikey_sets(self.load_negative_mode_spectra(), self.split_fraction)
            save_spectra(negative_training_spectra, self.negative_training_spectra_file)
            save_spectra(negative_validation_spectra, self.negative_validation_spectra_file)
            save_spectra(negative_testing_spectra, self.negative_testing_spectra_file)
        print(f"negative split \n "
              f"Train: {len(negative_training_spectra)} \n "
              f"Validation: {len(negative_validation_spectra)} \n "
              f"Test: {len(negative_testing_spectra)}")
        return negative_training_spectra, negative_validation_spectra, negative_testing_spectra

    def load_both_mode_train_split(self):
        positive_training_spectra, positive_validation_spectra, positive_testing_spectra = \
            self.load_positive_train_split()
        negative_training_spectra, negative_validation_spectra, negative_testing_spectra = \
            self.load_negative_mode_spectra()
        both_training_spectra = positive_training_spectra + negative_training_spectra
        both_validatation_spectra = positive_validation_spectra + negative_validation_spectra
        both_test_spectra = positive_testing_spectra + negative_testing_spectra
        return both_training_spectra, both_validatation_spectra, both_test_spectra
