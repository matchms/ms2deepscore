""" Data generators for training/inference with siamese Keras model.
"""
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

# Set random seed for reproducibility
np.random.seed(42)


class DataGeneratorAllInchikeys(Sequence):
    """Generates data for training a siamese Keras model
    """
    def __init__(self, spectrums_binned_dicts: list, training_ids: list,
                 y_true: np.ndarray, inchikey_mapping: pd.DataFrame, batch_size: int = 32,
                 num_turns: int = 1,
                 peak_scaling: float = 0.5,
                 dim: tuple = (10000,1), shuffle: bool = True, ignore_equal_pairs: bool = True,
                 inchikeys_array: np.ndarray = None,
                 same_prob_bins: list = [(0, 0.5), (0.5, 1)],
                 augment_peak_removal: dict = {"max_removal": 0.2, "max_intensity": 0.2},
                 augment_intensity: float = 0.1):
        """Generates data for training a siamese Keras model.
        This generator will provide training data by picking each training InchiKey
        listed in *training_ids* num_turns times in every epoch. It will then randomly
        pick one the spectra corresponding to this InchiKey (if multiple) and pair it
        with a randomly chosen other spectrum that corresponds to a reference score
        as defined in same_prob_bins.

        Parameters
        ----------
        spectrums_binned_dicts
            TODO: Make this less nested and clearer what I should exactly pass.
            List of dictionaries with the binned peak positions and intensities.
        training_ids
            List of IDs from the spectrums_binned_dicts list to use for training
        y_true
            TODO: In same order as what?
            Array of reference similarity scores (=labels).
        inchikey_mapping
            TODO: Needs documentation
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey during each epoch. Default=1
        peak_scaling
            Scale all peak intensities by power pf peak_scaling. Default is 0.5.
        dim
            Input vector dimension. Default=(10000,1)
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        ignore_equal_pairs
            Set to True to ignore pairs of two identical spectra. Default=True
        inchikeys_array
            TODO: Needs documentation
            TODO: I don't think it can be None?
        same_prob_bins
            TODO: Needs documentation
        augment_peak_removal
            TODO: Maybe have two parameters instead of a dictionary?
            Dictionary with two parameters. max_removal specifies the maximum amount
            of peaks to be removed (random fraction between 0 and max_removal), and
            max_intensity specifying that only peaks < max_intensity will be removed.
        augment_intensity
            Change peak intensities by a random number between 0 and augment_intensity.
            Default=0.1, which means that intensities are multiplied by 1+- a random
            number within [0, 0.1].

        """
        self.spectrums_binned_dicts = spectrums_binned_dicts
        self.score_array = y_true
        self.score_array[np.isnan(y_true)] = 0
        self.training_ids = training_ids
        self.dim = dim
        self.batch_size = batch_size
        self.num_turns = num_turns #number of go's through all IDs
        self.peak_scaling = peak_scaling
        self.shuffle = shuffle
        self.ignore_equal_pairs = ignore_equal_pairs
        self.on_epoch_end()
        self.inchikey_mapping = inchikey_mapping
        self.inchikeys_array = inchikeys_array
        self.same_prob_bins = same_prob_bins
        self.augment_peak_removal = augment_peak_removal
        self.augment_intensity = augment_intensity

    def __len__(self):
        """Denotes the number of batches per epoch"""
        # TODO: this means we don't see all data every epoch, because the last half-empty batch
        #  is omitted. I guess that is expected behavior?
        return int(self.num_turns) * int(np.floor(len(self.training_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Go through all indexes
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Select subset of IDs
        # TODO: Filter out function: select_batch_ids
        this_batch_ids = []
        for index in indexes:
            first_id_in_batch = self.training_ids[index]

            # TODO: Needs documentation
            prob_bins = self.same_prob_bins[np.random.choice(np.arange(len(self.same_prob_bins)))]

            # TODO: Factor out a function with documentation
            extend_range = 0
            while extend_range < 0.4:
                idx = np.where((self.score_array[first_id_in_batch, self.training_ids] > prob_bins[0] - extend_range)
                               & (self.score_array[first_id_in_batch, self.training_ids] <= prob_bins[1] + extend_range))[0]
                if self.ignore_equal_pairs:
                    idx = idx[idx != first_id_in_batch]
                if len(idx) > 0:
                    ID2 = self.training_ids[np.random.choice(idx)]
                    break
                extend_range += 0.1

            # TODO: Factor out a function with documentation
            if len(idx) == 0:
                #print(f"No matching pair found for score within {(prob_bins[0]-extend_range):.2f} and {(prob_bins[1]+extend_range):.2f}")
                #print(f"ID1: {ID1}")
                second_highest_id = self.score_array[first_id_in_batch, np.array([x for x in self.training_ids if x != first_id_in_batch])].argmax()
                if second_highest_id > first_id_in_batch:
                    second_highest_id += 1
                ID2 = self.training_ids[second_highest_id]
                #print(f"Picked ID2: {ID2}")

            this_batch_ids.append((first_id_in_batch, ID2))

        # Generate data
        X, y = self.__data_generation(this_batch_ids)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.tile(np.arange(len(self.training_ids)), int(self.num_turns))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_augmentation(self, document_dict):
        """Data augmentation.
            TODO: Needs documentation
            TODO: What is document_dict ?
        """
        idx = np.array([x for x in document_dict.keys()])
        values = np.array([x for x in document_dict.values()])
        if self.augment_peak_removal:
            # TODO: Factor out function with documentation + example?
            indices_select = np.where(values < self.augment_peak_removal["max_intensity"])[0]
            removal_part = np.random.random(1) * self.augment_peak_removal["max_removal"]
            indices_select = np.random.choice(indices_select,
                                              int(np.ceil((1 - removal_part)*len(indices_select))))
            indices = np.concatenate((indices_select,
                                      np.where(values >= self.augment_peak_removal["max_intensity"])[0]))
            if len(indices) > 0:
                idx = idx[indices]
                values = values[indices]
        if self.augment_intensity:
            # TODO: Factor out function with documentation + example?
            values = (1 - self.augment_intensity * 2 * (np.random.random(values.shape) - 0.5)) * values
        return idx, values

    def __data_generation(self, batch_ids):
        """Generates data containing batch_size samples"""
        # Initialization
        X = [np.zeros((self.batch_size, self.dim)) for i in range(2)]
        y = np.zeros((self.batch_size,))

        # Generate data
        for i_batch, pair in enumerate(batch_ids):
            for i_pair, id in enumerate(pair):
                inchikey = self.inchikey_mapping.loc[id]["inchikey"]
                spectrum_id = np.random.choice(np.where(self.inchikeys_array == inchikey)[0])
                idx, values = self.__data_augmentation(self.spectrums_binned_dicts[spectrum_id])
                X[i_pair][i_batch, idx] = values ** self.peak_scaling

            y[i_batch] = self.score_array[pair[0], pair[1]]

            # TODO: Add documentation. This means if there is no label we just add a random one
            #  right? Maybe it would be nicer to not include these (which would mean completely
            #  rearranging everything)
            if np.isnan(y[i_batch]):
                y[i_batch] = np.random.random(1)
        return X, y


class DataGenerator_all(Sequence):
    """Generates data for training a siamese Keras model
    """
    def __init__(self, spectrums_binned_dicts: list, list_IDs: list, list_score_IDs: list,
                 score_array: np.ndarray = None, batch_size: int = 32, num_turns: int = 1,
                 peak_scaling: float = 0.5,
                 dim: tuple = (10000,1), shuffle: bool = True, ignore_equal_pairs: bool = True,
                 inchikeys_array: np.ndarray = None, inchikey_mapping: pd.DataFrame = None,
                 same_prob_bins: list = [(0, 0.5), (0.5, 1)],
                 augment_peak_removal: dict = {"max_removal": 0.2, "max_intensity": 0.2},
                 augment_intensity: float = 0.1):
        """Generates data for training a siamese Keras model.
        This generator will provide training data by picking each training spectrum
        listed in *list_IDs* num_turns times in every epoch and pairing it with a randomly chosen
        other spectrum that corresponds to a reference score as defined in same_prob_bins.

        Parameters
        ----------
        spectrums_binned_dicts
            List of dictionaries with the binned peak positions and intensities.
        list_IDs
            List of IDs from the spectrums_binned_dicts list to use for training
        score_array
            Array of reference similarity scores (=labels).
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey during each epoch. Default=1
        peak_scaling
            Scale all peak intensities by power pf peak_scaling. Default is 0.5.
        dim
            Input vector dimension. Default=(10000,1)
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        ignore_equal_pairs
            Set to True to ignore pairs of two identical spectra. Default=True
        inchikeys_array
        inchikey_mapping
        same_prob_bins
        augment_peak_removal
            Dictionary with two parameters. max_removal specifies the maximum amount
            of peaks to be removed (random fraction between 0 and max_removal), and
            max_intensity specifying that only peaks < max_intensity will be removed.
        augment_intensity
            Change peak intensities by a random number between 0 and augment_intensity.
            Default=0.1, which means that intensities are multiplied by 1+- a random
            number within [0, 0.1].

        """
        self.spectrums_binned_dicts = spectrums_binned_dicts
        assert score_array is not None, "needs score array"
        self.score_array = score_array
        self.score_array[np.isnan(score_array)] = 0
        self.list_IDs = list_IDs
        self.list_score_IDs = list_score_IDs
        self.dim = dim
        self.batch_size = batch_size
        self.num_turns = num_turns #number of go's through all IDs
        self.peak_scaling = peak_scaling
        self.shuffle = shuffle
        self.ignore_equal_pairs = ignore_equal_pairs
        self.on_epoch_end()
        assert inchikey_mapping is not None, "needs inchikey mapping"
        self.inchikey_mapping = inchikey_mapping
        self.inchikeys_array = inchikeys_array
        self.same_prob_bins = same_prob_bins
        self.augment_peak_removal = augment_peak_removal
        self.augment_intensity = augment_intensity

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(self.num_turns) * int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Go through all indexes
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Select subset of IDs
        list_IDs_temp = []
        for index in indexes:
            ID1 = self.list_IDs[index]
            prob_bins = self.same_prob_bins[np.random.choice(np.arange(len(self.same_prob_bins)))]
            inchikey1 = self.inchikeys_array[ID1]
            #ID1_score = int(self.inchikey_mapping[self.inchikey_mapping == inchikey1].index[0])
            ID1_score = self.inchikey_mapping.values[np.where(self.inchikey_mapping.inchikey == inchikey1)][0][0]

            extend_range = 0
            while extend_range < 0.4:
                idx = np.where((self.score_array[ID1_score, self.list_score_IDs] > prob_bins[0] - extend_range)
                               & (self.score_array[ID1_score, self.list_score_IDs] <= prob_bins[1] + extend_range))[0]
                #if self.ignore_equal_pairs:
                #    idx = idx[idx != ID1]
                if len(idx) > 0:
                    ID2_score = self.list_score_IDs[np.random.choice(idx)]
                    break
                extend_range += 0.1

            if len(idx) == 0:
                #print(f"No matching pair found for score within {(prob_bins[0]-extend_range):.2f} and {(prob_bins[1]+extend_range):.2f}")
                #print(f"ID1: {ID1}")
                second_highest_id = self.score_array[ID1_score, np.array([x for x in self.list_score_IDs if x != ID1_score])].argmax()
                if second_highest_id > ID1_score:
                    second_highest_id += 1
                ID2_score = self.list_score_IDs[second_highest_id]
                #print(f"Picked ID2: {ID2}")

            # Get ID2 based on InchiKey
            inchikey2 = self.inchikey_mapping.loc[ID2_score]["inchikey"]
            idx2 = np.where(self.inchikeys_array == inchikey2)[0]
            if len(idx2) > 1 and self.ignore_equal_pairs:
                ID2 = np.random.choice([x for x in idx2 if x != ID1])
            else:
                ID2 = np.random.choice(idx2)
            list_IDs_temp.append((ID1, ID2, ID1_score, ID2_score))

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.tile(np.arange(len(self.list_IDs)), int(self.num_turns))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_augmentation(self, document_dict):
        """Data augmentation."""
        idx = np.array([x for x in document_dict.keys()])
        values = np.array([x for x in document_dict.values()])
        if self.augment_peak_removal:
            indices_select = np.where(values < self.augment_peak_removal["max_intensity"])[0]
            removal_part = np.random.random(1) * self.augment_peak_removal["max_removal"]
            indices_select = np.random.choice(indices_select,
                                              int(np.ceil((1 - removal_part)*len(indices_select))))
            indices = np.concatenate((indices_select,
                                      np.where(values >= self.augment_peak_removal["max_intensity"])[0]))
            if len(indices) > 0:
                idx = idx[indices]
                values = values[indices]
        if self.augment_intensity:
            values = (1 - self.augment_intensity * 2 * (np.random.random(values.shape) - 0.5)) * values
        return idx, values


    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X = [np.zeros((self.batch_size, self.dim)) for i in range(2)]
        y = np.zeros((self.batch_size,))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            ID1, ID2, ID1_score, ID2_score = IDs
            ref_idx, ref_values = self.__data_augmentation(self.spectrums_binned_dicts[ID1])

            # Add scaled weight to right bins
            X[0][i, ref_idx] = ref_values ** self.peak_scaling
            query_idx, query_values = self.__data_augmentation(self.spectrums_binned_dicts[ID2])

            # Add scaled weight to right bins
            X[1][i, query_idx] = query_values ** self.peak_scaling
            y[i] = self.score_array[ID1_score, ID2_score]
            if np.isnan(y[i]):
                y[i] = np.random.random(1)
        return X, y


# TODO: add symmetric nature of matrix to save factor 2
class DataGeneratorAllvsAll(Sequence):
    """Generates data for inference step (all-vs-all)"""
    def __init__(self, spectrums_binned_dicts, list_IDs, batch_size=32, dim=10000,
                 peak_scaling: float = 0.5,inchikey_mapping=None, inchikeys_array=None):
        """Generates data for prediction with a siamese Keras model

        Parameters
        ----------
        spectrums_binned_dicts
            List of dictionaries with the binned peak positions and intensities.
        list_IDs
            List of IDs from the spectrums_binned_dicts list to use for training
        batch_size
            Number of pairs per batch. Default=32.
        dim
            Input vector dimension. Default=(10000,1)
        peak_scaling
            Scale all peak intensities by power pf peak_scaling. Default is 0.5.
        inchikeys_array
        inchikey_mapping
        """
        self.spectrums_binned_dicts = spectrums_binned_dicts
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.peak_scaling = peak_scaling
        self.on_epoch_end()
        self.ID1 = 0
        self.ID2 = 0
        self.num_spectra = len(self.list_IDs)
        assert inchikey_mapping is not None, "needs inchikey mapping"
        self.inchikey_mapping = inchikey_mapping
        self.inchikeys_array = inchikeys_array

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.num_spectra / self.batch_size) * self.num_spectra)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Go through all x-y combinations
        pairs_list = []
        for i in range(self.batch_size):
            pairs_list.append((self.list_IDs[self.ID1], self.list_IDs[self.ID2]))
            self.ID2 += 1
            if self.ID2 >= self.num_spectra:
                self.ID2 = 0
                self.ID1 += 1
                if self.ID1 >= self.num_spectra:
                    self.ID1 = 0
                    print("job done...")
                break

        # Generate data
        X = self.__data_generation(pairs_list)

        return X

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.ID1 = 0
        self.ID2 = 0

    def __data_generation(self, pairs_list):
        """Generates data containing batch_size samples"""
        # Initialization
        X = [np.zeros((len(pairs_list), self.dim)) for i in range(2)]

        ID1 = pairs_list[0][0]
        inchikey_1 = self.inchikey_mapping.loc[ID1]["inchikey"]
        ID1_all = np.where(self.inchikeys_array == inchikey_1)[0][0]

        idx = np.array([x for x in spectrums_binned_dicts[ID1_all].keys()])
        X[0][:, idx] = np.array([x for x in spectrums_binned_dicts[ID1_all].values()]) ** self.peak_scaling

        # Generate data
        for i, IDs in enumerate(pairs_list):
            # Create binned spectrum vecotors and get similarity value
            inchikey_2 = self.inchikey_mapping.loc[IDs[1]]["inchikey"]
            ID2_all = np.where(self.inchikeys_array == inchikey_2)[0][0]
            idx = np.array([x for x in spectrums_binned_dicts[ID2_all].keys()])
            X[1][i, idx] = np.array([x for x in spectrums_binned_dicts[ID2_all].values()]) ** self.peak_scaling

        return X
