import json
import warnings
from datetime import datetime
from json import JSONEncoder
from typing import Optional
import numpy as np
from ms2deepscore.models.loss_functions import LOSS_FUNCTIONS


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
        patience:
            How long the model should keep training if validation does not improve
        loss_function:
            The loss function to use. The options can be found in models.loss_functions
        train_binning_layer
            Default is False in which case the model contains a first dense multi-group peak binning layer. If True a
            smart binning layer is used.
        train_binning_layer_group_size
            When a smart binning layer is used the group_size determines how many input bins are taken into
            one dense micro-network.
        train_binning_layer_output_per_group
            This sets the number of next layer bins each group_size group of inputs shares.
                batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey14 during each epoch. Default=1
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        same_prob_bins
            List of tuples that define ranges of the true label to be trained with
            equal frequencies. Default is set to [(0, 0.5), (0.5, 1)], which means
            that pairs with scores <=0.5 will be picked as often as pairs with scores
            > 0.5.
        average_pairs_per_bin:
            The aimed average number of pairs of spectra per spectrum in each bin.
        max_pairs_per_bin:
            The max_pairs_per_bin is used to reduce memory load.
            Since some spectra will have less than the average_pairs_per_bin, we can compensate by selecting more pairs for
            other spectra in this bin. For each spectrum initially max_pairs_per_bin is selected.
            If the max_oversampling_rate is too low, no good division can be created for the spectra.
            If the max_oversampling_rate is high the memory load on your system will be higher.
            If None, all pairs will be initially stored.
        include_diagonal:
            determines if a spectrum can be matched against itself when selection pairs.
        random_seed:
            The random seed to use for selecting compound pairs. Default is None.
        fingerprint_type:
            The fingerprint type that should be used for tanimoto score calculations.
        fingerprint_nbits:
            The number of bits to use for the fingerprint.
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
    def __init__(self, **settings):
        # model structure
        self.base_dims = (2000, 2000, 2000)
        self.embedding_dim = 400
        self.ionisation_mode = "positive"

        # additional model structure options
        self.train_binning_layer: bool = False
        self.train_binning_layer_group_size: int = 20
        self.train_binning_layer_output_per_group: int = 2

        # training settings
        self.dropout_rate = 0.0
        self.learning_rate = 0.00025
        self.epochs = 250
        self.patience = 20
        self.loss_function = "mse"
        self.weighting_factor = 0

        # Folder names for storing
        self.model_file_name = "ms2deepscore_model.pt"
        self.history_plot_file_name = "history.svg"
        self.time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # tensorization settings
        self.min_mz = 10
        self.max_mz = 1000
        self.mz_bin_width = 0.1
        self.intensity_scaling = 0.5
        self.additional_metadata = []

        # Data generator settings
        self.batch_size = 32
        self.num_turns = 1
        # todo shuffle and use fixed set can be removed, right? Since we dont use the datagenerator for val data.
        self.shuffle = True
        self.use_fixed_set = False

        # Compound pairs selection settings
        self.average_pairs_per_bin = 20
        self.max_pairs_per_bin = 100
        self.same_prob_bins = np.array([(x / 10, x / 10 + 0.1) for x in range(0, 10)])
        self.include_diagonal: bool = True
        self.random_seed: Optional[int] = None

        # Tanimioto score setings
        self.fingerprint_type: str = "daylight"
        self.fingerprint_nbits: int = 2048

        # Data augmentation
        self.augment_removal_max = 0.2
        self.augment_removal_intensity = 0.2
        self.augment_intensity = 0.2
        self.augment_noise_max = 10
        self.augment_noise_intensity = 0.02

        if settings:
            for key, value in settings.items():
                if hasattr(self, key):
                    if not isinstance(value, type(getattr(self, key))) and not getattr(self, key) is None:
                        raise TypeError(f"An unexpected type is given for the setting: {key}. "
                                        f"The expected type is {type(getattr(self, key))}, "
                                        f"the type given is {type(value)}, the value given is {value}")
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown setting: {key}")

        self.validate_settings()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def validate_settings(self):
        assert self.ionisation_mode in ("positive", "negative", "both")
        assert 0.0 <= self.augment_removal_max <= 1.0, "Expected value within [0,1]"
        assert 0.0 <= self.augment_removal_intensity <= 1.0, "Expected value within [0,1]"
        if self.use_fixed_set and self.shuffle:
            warnings.warn('When using a fixed set, data will not be shuffled')
        if self.random_seed is not None:
            assert isinstance(self.random_seed, int), "Random seed must be integer number."
        if self.loss_function.lower() not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss function. Must be one of: {LOSS_FUNCTIONS.keys()}")

    def number_of_bins(self):
        return int((self.max_mz - self.min_mz) / self.mz_bin_width)

    def get_dict(self):
        """returns a dictionary representation of the settings"""
        return self.__dict__

    def save_to_file(self, file_path):
        class NumpyArrayEncoder(JSONEncoder):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return JSONEncoder.default(self, o)
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.__dict__, file, indent=4, cls=NumpyArrayEncoder)


class SettingsEmbeddingEvaluator:
    """Contains all the settings used for training a EmbeddingEvaluator model.

    mini_batch_size
        Defines the actual trainig batch size after which the model weights are optimized.
    """
    def __init__(self, **settings):
        self.evaluator_distribution_size = 1000
        self.evaluator_num_filters = 48
        self.evaluator_depth = 3
        self.evaluator_kernel_size = 20
        self.random_seed: Optional[int] = None

        # Training settings
        self.mini_batch_size = 100
        self.batches_per_iteration = 1000
        self.learning_rate = 0.0001
        self.num_epochs = 5

        if settings:
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown setting: {key}")

    def get_dict(self):
        """returns a dictionary representation of the settings"""
        return self.__dict__
