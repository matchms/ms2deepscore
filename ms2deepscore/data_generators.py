""" Data generators for training/inference with siamese Keras model.
"""
import numpy as np
import pandas as pd
from typing import List
from tensorflow.keras.utils import Sequence
from ms2deepscore import BinnedSpectrum

# Set random seed for reproducibility
np.random.seed(42)


class DataGeneratorAllSpectrums(Sequence):
    """Generates data for training a siamese Keras model

    This generator will provide training data by picking each training spectrum
    listed in *spectrum_ids* num_turns times in every epoch and pairing it with a randomly chosen
    other spectrum that corresponds to a reference score as defined in same_prob_bins.
    """
    def __init__(self, spectrums_binned: List[BinnedSpectrum], spectrum_ids: list,
                 score_array: np.ndarray, inchikey_score_mapping: np.ndarray,
                 dim: int, **settings):
        """Generates data for training a siamese Keras model.

        Parameters
        ----------
        spectrums_binned
            List of BinnedSpectrum objects with the binned peak positions and intensities.
        spectrum_ids
            List of IDs from spectrums_binned to use for training.
        score_array
            2D-array of reference similarity scores (=labels).
            Must be symmetric (score_array[i,j] == score_array[j,i]) and in the
            same order as inchikey_score_mapping.
        inchikey_score_mapping
            Array of all unique inchikeys. Must be in the same order as the scores
            in scores_array.
        dim
            Input vector dimension.

        As part of **settings, defaults for the following parameters can be set:
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey during each epoch. Default=1.
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        ignore_equal_pairs
            Set to True to ignore pairs of two identical spectra. Default=True
        same_prob_bins
            List of tuples that define ranges of the true label to be trained with
            equal frequencies. Default is set to [(0, 0.5), (0.5, 1)], which means
            that pairs with scores <=0.5 will be picked as often as pairs with scores
            > 0.5.
        augment_removal_max
            Maximum fraction of peaks (if intensity < below augment_removal_intensity)
            to be removed randomly. Default is set to 0.2, which means that between
            0 and 20% of all peaks with intensities < augment_removal_intensity
            will be removed.
        augment_removal_intensity
            Specifying that only peaks with intensities < max_intensity will be removed.
        augment_intensity
            Change peak intensities by a random number between 0 and augment_intensity.
            Default=0.1, which means that intensities are multiplied by 1+- a random
            number within [0, 0.1].

        """
        assert score_array.shape[0] == score_array.shape[1] == len(inchikey_score_mapping), \
            f"Expected score_array of size {len(inchikey_score_mapping)}x{len(inchikey_score_mapping)}."
        self.spectrums_binned = spectrums_binned
        self.score_array = score_array
        #self.score_array[np.isnan(score_array)] = 0
        self.spectrum_ids = spectrum_ids
        self.inchikey_ids = self._exclude_nans(np.arange(score_array.shape[0]))
        self.inchikey_score_mapping = inchikey_score_mapping
        self.inchikeys_all = np.array([x.get("inchikey") for x in spectrums_binned])
        # TODO: add check if all inchikeys are present (should fail for missing ones)
        self.dim = dim

        # Set all other settings to input (or otherwise to defaults):
        self._set_generator_parameters(**settings)

        self.on_epoch_end()

    def _set_generator_parameters(self, **settings):
        """Set parameter for data generator. Use below listed defaults unless other
        input is provided.

        Parameters
        ----------
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey during each epoch. Default=1
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        ignore_equal_pairs
            Set to True to ignore pairs of two identical spectra. Default=True
        same_prob_bins
            List of tuples that define ranges of the true label to be trained with
            equal frequencies. Default is set to [(0, 0.5), (0.5, 1)], which means
            that pairs with scores <=0.5 will be picked as often as pairs with scores
            > 0.5.
        augment_removal_max
            Maximum fraction of peaks (if intensity < below augment_removal_intensity)
            to be removed randomly. Default is set to 0.2, which means that between
            0 and 20% of all peaks with intensities < augment_removal_intensity
            will be removed.
        augment_removal_intensity
            Specifying that only peaks with intensities < max_intensity will be removed.
        augment_intensity
            Change peak intensities by a random number between 0 and augment_intensity.
            Default=0.1, which means that intensities are multiplied by 1+- a random
            number within [0, 0.1].
        """
        defaults = dict(
            batch_size=32,
            num_turns=1,
            ignore_equal_pairs=True,
            shuffle=True,
            same_prob_bins=[(0, 0.5), (0.5, 1)],
            augment_removal_max= 0.3,
            augment_removal_intensity=0.2,
            augment_intensity=0.4,
        )

        # Set default parameters or replace by **settings input
        for key in defaults:
            if key in settings:
                print("The value for {} is set from {} (default) to {}".format(key, defaults[key],
                                                                              settings[key]))
            else:
                settings[key] = defaults[key]
        assert 0.0 <= settings["augment_removal_max"] <= 1.0, "Expected value within [0,1]"
        assert 0.0 <= settings["augment_removal_intensity"] <= 1.0, "Expected value within [0,1]"
        self.settings = settings

    def __len__(self):
        """Denotes the number of batches per epoch"""
        # TODO: this means we don't see all data every epoch, because the last half-empty batch
        #  is omitted. I guess that is expected behavior? --> Yes, with the shuffling in each epoch that seem OK to me (and makes the code easier).
        return int(self.num_turns) * int(np.floor(len(self.spectrum_ids) / self.settings["batch_size"]))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Go through all indexes
        indexes_spectrums = self.indexes[index*self.settings["batch_size"]:(index+1)*self.settings["batch_size"]]

        # Select subset of IDs
        # TODO: Filter out function: select_batch_ids
        spectrum_inchikey_ids_batch = []
        same_prob_bins = self.settings["same_prob_bins"]
        for index in indexes_spectrums:
            spectrum_id1 = self.spectrum_ids[index]
            inchikey_id1 = np.where(self.inchikey_score_mapping == self.inchikeys_all[spectrum_id1])[0]

            # Randomly pick the desired target score range and pick matching ID
            target_score_range = same_prob_bins[np.random.choice(np.arange(len(same_prob_bins)))]
            inchikey_id2 = self.__find_match_in_range(inchikey_id1, target_score_range)
            inchikey2 = self.inchikey_score_mapping[inchikey_id2]
            spectrum_id2 = np.random.choice(np.where(self.inchikeys_all == inchikey2)[0])

            spectrum_inchikey_ids_batch.append([(spectrum_id1, inchikey_id1), (spectrum_id2, inchikey_id2)])

        # Generate data
        X, y = self.__data_generation(spectrum_inchikey_ids_batch)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.tile(np.arange(len(self.spectrum_ids)), int(self.settings["num_turns"]))
        if self.settings["shuffle"] == True:
            np.random.shuffle(self.indexes)

    def _exclude_nans(self, inchikey_ids):
        """Find nans in labels and return list of IDs to be excluded."""
        find_nans = np.where(np.isnan(self.score_array))[0]
        if find_nans.shape[0] > 0:
            print(f"{find_nans.shape[0]} nans among {len(self.y_true)} labels will be excluded.")
        return [x for x in inchikey_ids if x not in list(find_nans)]

    def __find_match_in_range(self, inchikey_id1, target_score_range, max_range=0.4):
        """Randomly pick ID for a pair with inchikey_id1 that has a score in
        target_score_range. When no such score exists, iteratively widen the range
        in steps of 0.1 until a max of max_range. If still no match is found take
        a random ID.

        Parameters
        ----------
        inchikey_id1
            ID of inchikey to be paired up with another compound within target_score_range.
        target_score_range
            lower and upper bound of label (score) to find an ID of.
        """
        # Part 1 - find match within range (or expand range iteratively)
        extend_range = 0
        low, high = target_score_range
        while extend_range < max_range:
            print(inchikey_id1)
            idx = np.where((self.score_array[inchikey_id1, self.inchikey_ids] > low - extend_range)
                           & (self.score_array[inchikey_id1, self.inchikey_ids] <= high + extend_range))[0]
            if self.settings["ignore_equal_pairs"]:
                idx = idx[idx != inchikey_id1]
            if len(idx) > 0:
                inchikey_id2 = self.inchikey_ids[np.random.choice(idx)]
                break
            extend_range += 0.1

        # Part 2 - if still no match is found take the 2nd-highest score ID
        if len(idx) == 0:
            second_highest_id = self.score_array[inchikey_id1, np.array([x for x in self.inchikey_ids if x != inchikey_id1])].argmax()
            if second_highest_id > inchikey_id1:
                second_highest_id += 1
            inchikey_id2 = self.inchikey_ids[second_highest_id]

        return inchikey_id2

    def _data_augmentation(self, spectrum_binned):
        """Data augmentation.

        Parameters
        ----------
        spectrum_binned
            Dictionary with the binned peak positions and intensities.
        """
        idx = np.array([int(x) for x in spectrum_binned.keys()])
        values = np.array([x for x in spectrum_binned.values()])
        if self.settings["augment_removal_max"] or self.settings["augment_removal_intensity"]:
            # TODO: Factor out function with documentation + example?
            indices_select = np.where(values < self.settings["augment_removal_max"])[0]
            removal_part = np.random.random(1) * self.settings["augment_removal_max"]
            indices_select = np.random.choice(indices_select,
                                              int(np.ceil((1 - removal_part)*len(indices_select))))
            indices = np.concatenate((indices_select,
                                      np.where(values >= self.settings["augment_removal_intensity"])[0]))
            if len(indices) > 0:
                idx = idx[indices]
                values = values[indices]
        if self.settings["augment_intensity"]:
            # TODO: Factor out function with documentation + example?
            values = (1 - self.settings["augment_intensity"] * 2 * (np.random.random(values.shape) - 0.5)) * values
        return idx, values

    def __data_generation(self, spectrum_inchikey_ids_batch):
        """Generates data containing batch_size samples"""
        # Initialization
        X = [np.zeros((self.settings["batch_size"], self.dim)) for i in range(2)]
        y = np.zeros((self.settings["batch_size"],))

        # Generate data
        for i_batch, pair in enumerate(spectrum_inchikey_ids_batch):
            for i_pair, spectrum_inchikey_id in enumerate(pair):
                idx, values = self._data_augmentation(self.spectrums_binned[spectrum_inchikey_id[0]].peaks)
                X[i_pair][i_batch, idx] = values

            y[i_batch] = self.score_array[pair[0][1], pair[1][1]]

        return X, y


class DataGeneratorAllInchikeys(DataGeneratorAllSpectrums):
    """Generates data for training a siamese Keras model

    This generator will provide training data by picking each training InchiKey
    listed in *inchikey_ids* num_turns times in every epoch. It will then randomly
    pick one the spectra corresponding to this InchiKey (if multiple) and pair it
    with a randomly chosen other spectrum that corresponds to a reference score
    as defined in same_prob_bins.
    """
    def __init__(self, spectrums_binned: List[BinnedSpectrum], score_array: np.ndarray,
                 inchikey_ids: list, inchikey_score_mapping: np.ndarray,
                 inchikeys_all: np.ndarray, dim: int, **settings):
        """Generates data for training a siamese Keras model.

        Parameters
        ----------
        spectrums_binned
            List of BinnedSpectrum objects with the binned peak positions and intensities.
        score_array
            2D-array of reference similarity scores (=labels).
            Must be symmetric (score_array[i,j] == score_array[j,i]) and in the
            same order as inchikey_score_mapping.
        inchikey_ids
            List of IDs from unique_inchikeys to use for training.
        inchikey_score_mapping
            Array of all unique inchikeys. Must be in the same order as the scores
            in scores_array.
        inchikeys_all
            Array of all inchikeys. Must be in the same order as spectrums_binned.
        dim
            Input vector dimension.
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey during each epoch. Default=1
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        ignore_equal_pairs
            Set to True to ignore pairs of two identical spectra. Default=True

        same_prob_bins
            List of tuples that define ranges of the true label to be trained with
            equal frequencies. Default is set to [(0, 0.5), (0.5, 1)], which means
            that pairs with scores <=0.5 will be picked as often as pairs with scores
            > 0.5.
        augment_removal_max
            Maximum fraction of peaks (if intensity < below augment_removal_intensity)
            to be removed randomly. Default is set to 0.2, which means that between
            0 and 20% of all peaks with intensities < augment_removal_intensity
            will be removed.
        augment_removal_intensity
            Specifying that only peaks with intensities < max_intensity will be removed.
        augment_intensity
            Change peak intensities by a random number between 0 and augment_intensity.
            Default=0.1, which means that intensities are multiplied by 1+- a random
            number within [0, 0.1].

        """
        assert score_array.shape[0] == score_array.shape[1] == len(inchikey_score_mapping), \
            f"Expected score_array of size {len(inchikey_score_mapping)}x{len(inchikey_score_mapping)}."
        assert len(inchikeys_all) == len(spectrums_binned), \
            "inchikeys_all and spectrums_binned must have the same dimension."
        self.spectrums_binned = spectrums_binned
        self.score_array = score_array
        #self.score_array[np.isnan(score_array)] = 0
        self.inchikey_ids = self._exclude_nans(inchikey_ids)
        self.inchikey_score_mapping = inchikey_score_mapping
        self.inchikeys_all = inchikeys_all
        self.dim = dim
        self._set_generator_parameters(**settings)
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        # TODO: this means we don't see all data every epoch, because the last half-empty batch
        #  is omitted. I guess that is expected behavior? --> Yes, with the shuffling in each epoch that seem OK to me (and makes the code easier).
        return int(self.settings["num_turns"]) * int(np.floor(len(self.inchikey_ids) / self.settings["batch_size"]))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Go through all indexes
        indexes_inchkeys = self.indexes[index*self.settings["batch_size"]:(index+1)*self.settings["batch_size"]]

        # Select subset of IDs
        # TODO: Filter out function: select_batch_ids
        inchikey_ids_batch = []
        same_prob_bins = self.settings["same_prob_bins"]
        for index in indexes_inchkeys:
            inchikey_id1 = self.inchikey_ids[index]

            # Randomly pick the desired target score range and pick matching ID
            target_score_range = same_prob_bins[np.random.choice(np.arange(len(same_prob_bins)))]
            inchikey_id2 = self.__find_match_in_range(inchikey_id1, target_score_range)

            inchikey_ids_batch.append((inchikey_id1, inchikey_id2))

        # Generate data
        X, y = self.__data_generation(inchikey_ids_batch)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.tile(np.arange(len(self.inchikey_ids)), int(self.settings["num_turns"]))
        if self.settings["shuffle"] == True:
            np.random.shuffle(self.indexes)

    def __find_match_in_range(self, inchikey_id1, target_score_range, max_range=0.4):
        """Randomly pick ID for a pair with inchikey_id1 that has a score in
        target_score_range. When no such score exists, iteratively widen the range
        in steps of 0.1 until a max of max_range. If still no match is found take
        a random ID.

        Parameters
        ----------
        inchikey_id1
            ID of inchikey to be paired up with another compound within target_score_range.
        target_score_range
            lower and upper bound of label (score) to find an ID of.
        """
        # Part 1 - find match within range (or expand range iteratively)
        extend_range = 0
        low, high = target_score_range
        while extend_range < max_range:
            idx = np.where((self.score_array[inchikey_id1, self.inchikey_ids] > low - extend_range)
                           & (self.score_array[inchikey_id1, self.inchikey_ids] <= high + extend_range))[0]
            if self.settings["ignore_equal_pairs"]:
                idx = idx[idx != inchikey_id1]
            if len(idx) > 0:
                inchikey_id2 = self.inchikey_ids[np.random.choice(idx)]
                break
            extend_range += 0.1

        # Part 2 - if still no match is found take the 2nd-highest score ID
        if len(idx) == 0:
            second_highest_id = self.score_array[inchikey_id1, np.array([x for x in self.inchikey_ids if x != inchikey_id1])].argmax()
            if second_highest_id > inchikey_id1:
                second_highest_id += 1
            inchikey_id2 = self.inchikey_ids[second_highest_id]

        return inchikey_id2

    def __data_generation(self, inchikey_ids_batch):
        """Generates data containing batch_size samples"""
        # Initialization
        X = [np.zeros((self.settings["batch_size"], self.dim)) for i in range(2)]
        y = np.zeros((self.settings["batch_size"],))

        # Generate data
        for i_batch, pair in enumerate(inchikey_ids_batch):
            for i_pair, inchikey_id in enumerate(pair):
                inchikey = self.inchikey_score_mapping[inchikey_id]
                spectrum_id = np.random.choice(np.where(self.inchikeys_all == inchikey)[0])
                idx, values = self._data_augmentation(self.spectrums_binned[spectrum_id].peaks)
                X[i_pair][i_batch, idx] = values

            y[i_batch] = self.score_array[pair[0], pair[1]]

        return X, y
