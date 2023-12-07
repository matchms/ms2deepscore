import json
import warnings
from datetime import datetime
from json import JSONEncoder
from typing import Optional
import numpy as np


class SettingsMS2Deepscore:
    """Contains all the settings used for training a MS2Deepscore model.

    Attributes:
        base_dims:
            The in between layers to be used. Default = (500, 500)
        embedding_dim:
            The dimension of the final embedding. Default = 200
        additional_metadata:
            Additional metadata that should be used in training the model. e.g. precursor_mz
        dropout_rate:
            The dropout rate that should be used during training
        learning_rate:
            The learning rate that should be used during training.
        epochs:
            The number of epochs that should be used during training.
        average_pairs_per_bin:
            The aimed average number of pairs of spectra per spectrum in each bin.
        max_pairs_per_bin:
            The max_pairs_per_bin is used to reduce memory load.
            Since some spectra will have less than the average_pairs_per_bin, we can compensate by selecting more pairs for
            other spectra in this bin. For each spectrum initially max_pairs_per_bin is selected.
            If the max_oversampling_rate is too low, no good division can be created for the spectra.
            If the max_oversampling_rate is high the memory load on your system will be higher.
            If None, all pairs will be initially stored.
        tanimoto_bins:
            The tanimoto score bins that should be used. Default is 10 bins equally spread between 0 and 1.
        include_diagonal:
            determines if a spectrum can be matched against itself when selection pairs.
        random_seed:
            The random seed to use for selecting compound pairs. Default is None.
        fingerprint_type:
            The fingerprint type that should be used for tanimoto score calculations.
        fingerprint_nbits:
            The number of bits to use for the fingerprint.
        """
    def __init__(self, settings=None):
        # model structure
        self.base_dims = (500, 500)
        self.embedding_dim = 200
        self.additional_metadata = ()
        self.ionisation_mode = "positive"

        # training settings
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.epochs = 150
        self.patience = 10

        # Generator settings
        self.batch_size = 32

        # Compound pairs selection settings
        self.average_pairs_per_bin = 20
        self.max_pairs_per_bin = 100
        self.tanimoto_bins: np.ndarray = np.array([(x / 10, x / 10 + 0.1) for x in range(0, 10)])
        self.include_diagonal: bool = True
        self.random_seed: Optional[int] = None

        # Tanimioto score setings
        self.fingerprint_type: str = "daylight"
        self.fingerprint_nbits: int = 2048

        # Folder names for storing
        self.binned_spectra_folder_name = "binned_spectra"
        self.model_file_name = "ms2deepscore_model.hdf5"
        self.history_file_name = "history.txt"
        self.history_plot_file_name = "history.svg"

        if settings:
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown setting: {key}")
        self.model_directory_name = self._create_model_directory_name()
        self.validate_settings()

    def validate_settings(self):
        assert self.ionisation_mode in ("positive", "negative", "both")

    def _create_model_directory_name(self):
        """Creates a directory name using metadata, it will contain the metadata, the binned spectra and final model"""
        binning_file_label = ""
        for metadata_generator in self.additional_metadata:
            binning_file_label += metadata_generator.metadata_field + "_"

        # Define a neural net structure label
        neural_net_structure_label = ""
        for layer in self.base_dims:
            neural_net_structure_label += str(layer) + "_"
        neural_net_structure_label += "layers"

        if self.embedding_dim:
            neural_net_structure_label += f"_{str(self.embedding_dim)}_embedding"
        time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_folder_file_name = f"{self.ionisation_mode}_mode_{binning_file_label}" \
                                 f"{neural_net_structure_label}_{time_stamp}"
        print(f"The model will be stored in the folder: {model_folder_file_name}")
        return model_folder_file_name

    def save_to_file(self, file_path):
        class NumpyArrayEncoder(JSONEncoder):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return JSONEncoder.default(self, o)
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.__dict__, file, indent=4, cls=NumpyArrayEncoder)


class GeneratorSettings:
    """
    Set parameters for data generator. Use below listed defaults unless other
    input is provided.

    Parameters
    ----------
    settings:
        A dictionary containing the settings that need to be changed. For parameters that are not given the default
        will be used.
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
        augment_noise_max
            Max number of 'new' noise peaks to add to the spectrum, between 0 to `augment_noise_max`
            of peaks are added.
        augment_noise_intensity
            Intensity of the 'new' noise peaks to add to the spectrum
        use_fixed_set
            Toggles using a fixed dataset, if set to True the same dataset will be generated each
            epoch. Default is False.
        random_seed
            Specify random seed for reproducible random number generation.
        additional_inputs
            Array of additional values to be used in training for e.g. ["precursor_mz", "parent_mass"]
    """
    def __init__(self, settings=None):
        self.batch_size = 32
        self.num_turns = 1
        self.ignore_equal_pairs = True
        self.shuffle = True
        self.same_prob_bins = [(0, 0.5), (0.5, 1)]
        self.augment_removal_max = 0.3
        self.augment_removal_intensity = 0.2
        self.augment_intensity = 0.4
        self.augment_noise_max = 10
        self.augment_noise_intensity = 0.01
        self.use_fixed_set = False
        self.random_seed = None
        if settings:
            for key, value in settings.items():
                if hasattr(self, key):
                    print(f"The value for {key} is set from {getattr(self, key)} (default) to {value}")
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown setting: {key}")
        self.validate_settings()

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def validate_settings(self):
        assert 0.0 <= self.augment_removal_max <= 1.0, "Expected value within [0,1]"
        assert 0.0 <= self.augment_removal_intensity <= 1.0, "Expected value within [0,1]"
        if self.use_fixed_set and self.shuffle:
            warnings.warn('When using a fixed set, data will not be shuffled')
        if self.random_seed is not None:
            assert isinstance(self.random_seed, int), "Random seed must be integer number."
