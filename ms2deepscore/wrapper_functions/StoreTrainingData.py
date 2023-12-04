import os
from typing import Generator, List
from matchms import Spectrum
from matchms.exporting import save_spectra
from matchms.importing import load_spectra

from ms2deepscore.train_new_model.split_positive_and_negative_mode import split_pos_and_neg
from ms2deepscore.train_new_model.validation_and_test_split import split_spectra_in_random_inchikey_sets


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
        os.makedirs(self.positive_negative_split_dir, exist_ok=True)

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
        return list(negative_mode_spectra)

    def split_and_save_positive_and_negative_spectra(self):
        assert not os.path.isfile(self.positive_mode_spectra_file), "the positive mode spectra file already exists"
        assert not os.path.isfile(self.negative_mode_spectra_file), "the negative mode spectra file already exists"
        positive_mode_spectra, negative_mode_spectra = split_pos_and_neg(
            load_spectra(self.spectra_file_name, metadata_harmonization=False))
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
            positive_training_spectra = load_spectra_as_list(self.positive_training_spectra_file)
            positive_validation_spectra = load_spectra_as_list(self.positive_validation_spectra_file)
            positive_testing_spectra = load_spectra_as_list(self.positive_testing_spectra_file)
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
            negative_training_spectra = load_spectra_as_list(self.negative_training_spectra_file)
            negative_validation_spectra = load_spectra_as_list(self.negative_validation_spectra_file)
            negative_testing_spectra = load_spectra_as_list(self.negative_testing_spectra_file)
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

    def load_train_val_data(self,
                            ionisation_mode: str):
        """Loads the train, val and test spectra for a specified mode."""
        if ionisation_mode == "positive":
            return self.load_positive_train_split()
        if ionisation_mode == "negative":
            return self.load_negative_train_split()
        if ionisation_mode == "both":
            return self.load_both_mode_train_split()
        raise ValueError("expected ionisation mode to be 'positive', 'negative' or 'both'")


def load_spectra_as_list(file_name) -> List[Spectrum]:
    spectra = load_spectra(file_name, metadata_harmonization=False)
    if isinstance(spectra, Generator):
        return list(spectra)
    return spectra