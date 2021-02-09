""" Data generators for training/inference with siamese Keras model.
"""
from typing import List, Iterator, NamedTuple
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from .typing import BinnedSpectrumType

# Set random seed for reproducibility
np.random.seed(42)


class SpectrumPair(NamedTuple):
    """
    Represents a pair of binned spectrums
    """
    spectrum1: BinnedSpectrumType
    spectrum2: BinnedSpectrumType


class DataGeneratorBase(Sequence):
    def __init__(self, binned_spectrums: List[BinnedSpectrumType],
                 reference_scores_df: pd.DataFrame, dim: int, **settings):
        """Base for data generator generating data for a siamese model.

        Parameters
        ----------
        binned_spectrums
            List of BinnedSpectrum objects with the binned peak positions and intensities.
        reference_scores_df
            Pandas DataFrame with reference similarity scores (=labels) for compounds identified
            by inchikeys (first 14 characters). Columns and index should be inchikeys, the value
            in a row x column depicting the similarity score for that pair. Must be symmetric
            (reference_scores_df[i,j] == reference_scores_df[j,i]) and column names should be
            identical to the index.
        dim
            Input vector dimension.

        As part of **settings, defaults for the following parameters can be set:
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey14 during each epoch. Default=1.
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

        self._validate_labels(reference_scores_df)

        # Set all other settings to input (or otherwise to defaults):
        self._set_generator_parameters(**settings)
        self.binned_spectrums = binned_spectrums
        self.reference_scores_df = self._exclude_nans_from_labels(reference_scores_df)
        self.reference_scores_df = self._transform_to_inchikey14(self.reference_scores_df)
        self._collect_and_validate_inchikeys()
        self.dim = dim

    def _collect_and_validate_inchikeys(self):
        """Collect all inchikeys14 (first 14 characters) of all binned_spectrums
        and check if all are present in the reference scores as well.
        """
        self.spectrum_inchikeys = np.array([s.get("inchikey")[:14] for s in self.binned_spectrums])
        for inchikey in np.unique(self.spectrum_inchikeys):
            assert inchikey in self.reference_scores_df.index, \
                "InChIKey in given spectrum not found in reference scores"

    @staticmethod
    def _validate_labels(reference_scores_df: pd.DataFrame):
        if set(reference_scores_df.index) != set(reference_scores_df.columns):
            raise ValueError("index and columns of reference_scores_df are not identical")

    @staticmethod
    def _transform_to_inchikey14(reference_scores_df: pd.DataFrame):
        """Transform index and column names from potential full InChIKeys to InChIKey14"""
        reference_scores_df.index = [x[:14] for x in reference_scores_df.index]
        reference_scores_df.columns = [x[:14] for x in reference_scores_df.columns]
        return reference_scores_df

    @staticmethod
    def _exclude_nans_from_labels(reference_scores_df: pd.DataFrame):
        """Exclude nans in reference_scores_df, exclude columns and rows if there is any NaN
        value"""
        clean_df = reference_scores_df.dropna(axis='rows')  # drop rows with any NaN
        clean_df = clean_df[clean_df.index]  # drop corresponding columns
        n_dropped = len(reference_scores_df) - len(clean_df)
        if n_dropped > 0:
            print(f"{n_dropped} nans among {len(reference_scores_df)} labels will be excluded.")
        return clean_df

    def _set_generator_parameters(self, **settings):
        """Set parameter for data generator. Use below listed defaults unless other
        input is provided.

        Parameters
        ----------
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey14 during each epoch. Default=1
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
            augment_noise_max=10,
            augment_noise_intensity=0.01,
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

    def _find_match_in_range(self, inchikey1, target_score_range, max_range=0.4):
        """Randomly pick ID for a pair with inchikey_id1 that has a score in
        target_score_range. When no such score exists, iteratively widen the range
        in steps of 0.1 until a max of max_range. If still no match is found take
        a random ID.
        Parameters
        ----------
        inchikey1
            Inchikey (first 14 characters) to be paired up with another compound within
            target_score_range.
        target_score_range
            lower and upper bound of label (score) to find an ID of.
        """
        # Part 1 - find match within range (or expand range iteratively)
        extend_range = 0
        low, high = target_score_range
        inchikey2 = None
        while extend_range < max_range:
            matching_inchikeys = self.reference_scores_df.index[
                (self.reference_scores_df[inchikey1] > low - extend_range)
                & (self.reference_scores_df[inchikey1] <= high + extend_range)]
            if self.settings["ignore_equal_pairs"]:
                matching_inchikeys = matching_inchikeys[matching_inchikeys != inchikey1]
            if len(matching_inchikeys) > 0:
                inchikey2 = np.random.choice(matching_inchikeys)
                break
            extend_range += 0.1

        # Part 2 - if still no match is found take the inchikey that has highest similarity score
        # with inchikey1
        # TODO: Why are we taking the highest score here? Shouldn't we take the one that is closest
        #  to the center of the bin?
        if not inchikey2:
            inchikey2 = self.reference_scores_df[inchikey1][self.reference_scores_df.index != inchikey1].idxmax()

        return inchikey2

    def __getitem__(self, batch_index: int):
        """Generate one batch of data"""
        spectrum_pairs = self._spectrum_pair_generator(batch_index)
        X, y = self.__data_generation(spectrum_pairs)
        return X, y

    def _data_augmentation(self, spectrum_binned):
        """Data augmentation.
        Parameters
        ----------
        spectrum_binned
            Dictionary with the binned peak positions and intensities.
        """
        idx = np.array([int(x) for x in spectrum_binned.keys()])
        values = np.array(list(spectrum_binned.values()))
        # Augmentation 1: peak removal (peaks < augment_removal_max)
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
        # Augmentation 2: Change peak intensities
        if self.settings["augment_intensity"]:
            # TODO: Factor out function with documentation + example?
            values = (1 - self.settings["augment_intensity"] * 2 * (np.random.random(values.shape) - 0.5)) * values
        # Augmentation 3: Add noise peaks
        if self.settings["augment_noise_max"] and self.settings["augment_noise_max"] > 0:
            n_noise_peaks = np.random.randint(0, self.settings["augment_noise_max"])
            idx_noise_peaks = np.random.randint(0, self.dim, n_noise_peaks)
            idx_noise_peaks = idx_noise_peaks[np.isin(idx_noise_peaks, idx, invert=True)]
            idx = np.concatenate((idx, idx_noise_peaks))
            values = np.concatenate((values,
                                     self.settings["augment_noise_intensity"] * np.random.random(len(idx_noise_peaks))))
        return idx, values

    def _get_spectrum_with_inchikey(self, inchikey: str) -> BinnedSpectrumType:
        """
        Get a random spectrum matching the `inchikey` argument. NB: A compound (identified by an
        inchikey) can have multiple measured spectrums in a binned spectrum dataset.
        """
        matching_spectrum_id = np.where(self.spectrum_inchikeys == inchikey)[0]
        return self.binned_spectrums[np.random.choice(matching_spectrum_id)]

    def __data_generation(self, spectrum_pairs: Iterator[SpectrumPair]):
        """Generates data containing batch_size samples"""
        # Initialization
        X = [np.zeros((self.settings["batch_size"], self.dim)) for i in range(2)]
        y = np.zeros((self.settings["batch_size"],))

        # Generate data
        for i_pair, pair in enumerate(spectrum_pairs):
            for i_spectrum, spectrum in enumerate(pair):
                idx, values = self._data_augmentation(spectrum.binned_peaks)
                X[i_spectrum][i_pair, idx] = values
            y[i_pair] = self.reference_scores_df[pair[0].get("inchikey")[:14]][pair[1].get("inchikey")[:14]]

        return X, y

    def _spectrum_pair_generator(self, batch_index: int) -> Iterator[SpectrumPair]:
        """
        Generator of spectrum pairs within a batch, inheriting classes should implement this.
        """
        raise NotImplementedError()


class DataGeneratorAllSpectrums(DataGeneratorBase):
    """Generates data for training a siamese Keras model
    This generator will provide training data by picking each training spectrum
    in binned_spectrums num_turns times in every epoch and pairing it with a randomly chosen
    other spectrum that corresponds to a reference score as defined in same_prob_bins.
    """
    def __init__(self, binned_spectrums: List[BinnedSpectrumType],
                 reference_scores_df: pd.DataFrame, dim: int, **settings):
        """Generates data for training a siamese Keras model.
        Parameters
        ----------
        binned_spectrums
            List of BinnedSpectrum objects with the binned peak positions and intensities.
        reference_scores_df
            Pandas DataFrame with reference similarity scores (=labels) for compounds identified
            by inchikeys. Columns and index should be inchikeys, the value in a row x column
            depicting the similarity score for that pair. Must be symmetric
            (reference_scores_df[i,j] == reference_scores_df[j,i]) and column names should be
            identical to the index.
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
        super().__init__(binned_spectrums, reference_scores_df, dim, **settings)
        self.reference_scores_df = self._exclude_not_selected_inchikeys(self.reference_scores_df)
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        # TODO: this means we don't see all data every epoch, because the last half-empty batch
        #  is omitted. I guess that is expected behavior? --> Yes, with the shuffling in each epoch that seem OK to me (and makes the code easier).
        return int(self.settings["num_turns"]) * int(np.floor(len(self.binned_spectrums) / self.settings["batch_size"]))

    def _spectrum_pair_generator(self, batch_index: int) -> Iterator[SpectrumPair]:
        """
        Generate spectrum pairs for batch. For each 'source' spectrum, get the inchikey and
        find an inchikey in the desired target score range. Then randomly get a spectrums for
        the maching inchikey.
        """
        same_prob_bins = self.settings["same_prob_bins"]
        batch_size = self.settings["batch_size"]
        indexes = self.indexes[batch_index*batch_size:(batch_index+1)*batch_size]
        for index in indexes:
            spectrum1 = self.binned_spectrums[index]
            inchikey1 = spectrum1.get("inchikey")[:14]

            # Randomly pick the desired target score range and pick matching ID
            target_score_range = same_prob_bins[np.random.choice(np.arange(len(same_prob_bins)))]
            inchikey2 = self._find_match_in_range(inchikey1, target_score_range)
            spectrum2 = self._get_spectrum_with_inchikey(inchikey2)
            yield SpectrumPair(spectrum1, spectrum2)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.tile(np.arange(len(self.binned_spectrums)), int(self.settings["num_turns"]))
        if self.settings["shuffle"]:
            np.random.shuffle(self.indexes)

    @staticmethod
    def _exclude_nans_from_labels(reference_scores_df: pd.DataFrame):
        """Exclude nans in reference_scores_df, exclude columns and rows if there is any NaN
        value"""
        clean_df = reference_scores_df.dropna(axis='rows')  # drop rows with any NaN
        clean_df = clean_df[clean_df.index]  # drop corresponding columns
        n_dropped = len(reference_scores_df) - len(clean_df)
        if n_dropped > 0:
            print(f"{n_dropped} nans among {len(reference_scores_df)} labels will be excluded.")
        return clean_df

    def _exclude_not_selected_inchikeys(self, reference_scores_df: pd.DataFrame) -> pd.DataFrame:
        """Exclude rows and columns of reference_scores_df for all InChIKeys which are not
        present in the binned_spectrums."""
        inchikeys_in_selection = {s.get("inchikey")[:14] for s in self.binned_spectrums}
        clean_df = reference_scores_df.loc[reference_scores_df.index.isin(inchikeys_in_selection),
                                 reference_scores_df.columns.isin(inchikeys_in_selection)]
        n_dropped = len(self.reference_scores_df) - len(clean_df)
        if n_dropped > 0:
            print(f"{len(clean_df)} out of {len(self.reference_scores_df)} InChIKeys found in selected spectrums.")
        return clean_df


class DataGeneratorAllInchikeys(DataGeneratorBase):
    """Generates data for training a siamese Keras model
    This generator will provide training data by picking each training InchiKey
    listed in *selected_inchikeys* num_turns times in every epoch. It will then randomly
    pick one the spectra corresponding to this InchiKey (if multiple) and pair it
    with a randomly chosen other spectrum that corresponds to a reference score
    as defined in same_prob_bins.
    """
    def __init__(self, binned_spectrums: List[BinnedSpectrumType], selected_inchikeys: list,
                 reference_scores_df: pd.DataFrame, dim: int, **settings):
        """Generates data for training a siamese Keras model.
        Parameters
        ----------
        binned_spectrums
            List of BinnedSpectrum objects with the binned peak positions and intensities.
        reference_scores_df
            Pandas DataFrame with reference similarity scores (=labels) for compounds identified
            by inchikeys. Columns and index should be inchikeys, the value in a row x column
            depicting the similarity score for that pair. Must be symmetric
            (reference_scores_df[i,j] == reference_scores_df[j,i]) and column names should be identical to the index.
        selected_inchikeys
            List of inchikeys to use for training.
        dim
            Input vector dimension.
        As part of **settings, defaults for the following parameters can be set:
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
        super().__init__(binned_spectrums, reference_scores_df, dim, **settings)
        self.reference_scores_df = self._data_selection(reference_scores_df, selected_inchikeys)
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        # TODO: this means we don't see all data every epoch, because the last half-empty batch
        #  is omitted. I guess that is expected behavior? --> Yes, with the shuffling in each epoch that seem OK to me (and makes the code easier).
        return int(self.settings["num_turns"]) * int(np.floor(len(self.reference_scores_df) / self.settings[
            "batch_size"]))

    def _spectrum_pair_generator(self, batch_index: int) -> Iterator[SpectrumPair]:
        """
        Generate spectrum pairs for batch. For each 'source' inchikey pick an inchikey in the
        desired target score range. Then randomly get spectrums for this pair of inchikeys.
        """
        same_prob_bins = self.settings["same_prob_bins"]
        batch_size = self.settings["batch_size"]
        # Go through all indexes
        indexes = self.indexes[batch_index * batch_size:(batch_index + 1) * batch_size]

        for index in indexes:
            inchikey1 = self.reference_scores_df.index[index]
            # Randomly pick the desired target score range and pick matching inchikey
            target_score_range = same_prob_bins[np.random.choice(np.arange(len(same_prob_bins)))]
            inchikey2 = self._find_match_in_range(inchikey1, target_score_range)
            spectrum1 = self._get_spectrum_with_inchikey(inchikey1)
            spectrum2 = self._get_spectrum_with_inchikey(inchikey2)
            yield SpectrumPair(spectrum1, spectrum2)

    @staticmethod
    def _data_selection(reference_scores_df, selected_inchikeys):
        """
        Select labeled data to generate from based on `selected_inchikeys`
        """
        return reference_scores_df.loc[selected_inchikeys, selected_inchikeys]

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.tile(np.arange(len(self.reference_scores_df)), int(self.settings["num_turns"]))
        if self.settings["shuffle"]:
            np.random.shuffle(self.indexes)
