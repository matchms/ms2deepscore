import json
import warnings
from pathlib import Path as _Path
from typing import Any, Dict
from datetime import datetime
from json import JSONEncoder
from typing import Optional
import numpy as np
from ms2deepscore.models.loss_functions import LOSS_FUNCTIONS
from ms2deepscore.utils import validate_bin_order


def _coerce_value(expected_example: Any, value: Any) -> Any:
    """
    Coerce `value` to the type suggested by `expected_example` (the current attribute's default).
    This is best-effort and conservative: if coercion fails, returns the original `value`.
    """
    try:
        # None stays None
        if value is None:
            return None

        expected_type = type(expected_example)

        # Common container/array cases
        if expected_type is tuple and isinstance(value, list):
            return tuple(value)
        if expected_type is set and isinstance(value, list):
            return set(value)
        # For numpy arrays, accept list (or tuple) and convert
        if isinstance(expected_example, np.ndarray) and isinstance(value, (list, tuple)):
            return np.array(value)

        # Path-like
        if expected_type is _Path and isinstance(value, str):
            return _Path(value)

        # Coercing boolean
        if expected_type is bool and isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "y"):
                return True
            elif low in ("false", "0", "no", "n"):
                return False
            raise ValueError(f"Invalid boolean value: {value}")

        if expected_type is int and isinstance(value, (float, str)):
            return int(value)

        if expected_type is float and isinstance(value, (int, str)):
            return float(value)

        if expected_type is _Path and isinstance(value, str):
            return _Path(value)

        # If expected is np.ndarray of shape (n,2) (e.g., same_prob_bins), tolerate list of lists
        if isinstance(expected_example, np.ndarray) and isinstance(value, list):
            arr = np.array(value)
            return arr

        # Already right type or compatible
        return value

    except Exception:
        # On any failure, return the original value
        return value


def _coerce_settings_dict(settings_in: Dict[str, Any], defaults_obj: Any) -> Dict[str, Any]:
    """
    Build a new dict with values coerced to the types implied by `defaults_obj`'s current attributes.
    Unknown keys are preserved (caller decides whether to accept/ignore them).
    """
    out = dict(settings_in)
    for k, v in list(out.items()):
        if hasattr(defaults_obj, k):
            out[k] = _coerce_value(getattr(defaults_obj, k), v)
    return out


class SettingsMS2Deepscore:
    """Contains all the settings used for training a MS2Deepscore model.

    Attributes:
        base_dims:
            The in between layers to be used. Default = (2000, 2000, 2000)
        embedding_dim:
            The dimension of the final embedding. Default = 400
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
            equal frequencies. Default is set to 10 bins of equal width between 0 and 1.
        average_pairs_per_bin:
            The aimed average number of pairs of spectra per spectrum in each bin.
        max_pairs_per_bin:
            The max_pairs_per_bin is used to reduce memory load.
            Since some spectra will have less than the average_pairs_per_bin, we can compensate by selecting more pairs for
            other spectra in this bin. For each spectrum initially max_pairs_per_bin is selected.
            If the max_pairs_per_bin is too low, no good division can be created for the spectra.
            If the max_pairs_per_bin is high the memory load on your system will be higher.
            If None, all pairs will be initially stored.
        include_diagonal:
            determines if a spectrum can be matched against itself when selection pairs.
        val_spectra_per_inchikey:
            Set number of spectra to pick per inchikey in the validation set. 
            The larger this number, the slower the validation loss computation.
            Default is set to 1.
        random_seed:
            The random seed to use for selecting compound pairs. Default is None.
        fingerprint_type:
            The fingerprint type that should be used for Tanimoto score calculations.
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
        additional_metadata
            Array of metadata entries (and their transformation) to be used in training.
            See `MetadatFeatureGenerator` for more information.
            Default is set to empty list.
        max_pair_resampling
            The maximum number a inchikey pair can be resampled. Resampling is done to balance inchikey pairs over
            the tanimoto scores. The minimum is 1, meaning that no resampling is performed.
        """
    def __init__(self, validate_settings=True, **settings):
        # model structure
        self.base_dims = (10000,)
        self.embedding_dim = 500
        self.ionisation_mode = "positive"
        self.activation_function = "relu"

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
        self.average_inchikey_sampling_count = 100
        self.max_inchikey_sampling = 110
        self.max_pairs_per_bin = 300
        self.same_prob_bins = np.array([(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
                                        (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (-0.01, 0.1)])
        self.include_diagonal = True
        self.val_spectra_per_inchikey = 1
        self.random_seed: Optional[int] = None
        self.max_pair_resampling = 10000000

        # Tanimioto score setings
        self.fingerprint_type: str = "daylight"
        self.fingerprint_nbits: int = 4096

        # Data augmentation
        self.augment_removal_max = 0.2
        self.augment_removal_intensity = 0.2
        self.augment_intensity = 0.2
        self.augment_noise_max = 10
        self.augment_noise_intensity = 0.02

        if settings:
            # coerce a copy against current defaults
            settings = _coerce_settings_dict(settings, self)

            for key, value in settings.items():
                if hasattr(self, key):
                    # after coercion, keep strict type check
                    if not isinstance(value, type(getattr(self, key))) and getattr(self, key) is not None:
                        raise TypeError(
                            f"An unexpected type is given for the setting: {key}. "
                            f"The expected type is {type(getattr(self, key))}, "
                            f"the type given is {type(value)}, the value given is {value}"
                        )
                    setattr(self, key, value)
                else:
                    if validate_settings:
                        raise ValueError(f"Unknown setting: {key}")
                    # keep legacy/unknown keys for backward-compat if validate_settings=False
                    setattr(self, key, value)

        if validate_settings:
            self.validate_settings()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def validate_settings(self):
        if self.ionisation_mode not in ("positive", "negative", "both"):
            raise ValueError("Expected ionisation mode to be 'positive' , 'negative', or 'both'.")
        if not (0.0 <= self.augment_removal_max <= 1.0) or (not 0.0 <= self.augment_removal_intensity <= 1.0):
            raise ValueError("Expected value within [0,1]")
        if self.use_fixed_set and self.shuffle:
            warnings.warn('When using a fixed set, data will not be shuffled')
        if (self.random_seed is not None) and not isinstance(self.random_seed, int):
            raise ValueError("Random seed must be integer number.")
        if self.loss_function.lower() not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss function. Must be one of: {LOSS_FUNCTIONS.keys()}")
        validate_bin_order(self.same_prob_bins)

    def number_of_bins(self):
        return int((self.max_mz - self.min_mz) / self.mz_bin_width)

    def get_dict(self):
        """returns a dictionary representation of the settings"""
        settings_dict = self.__dict__.copy()
        
        for key, value in settings_dict.items():
            if isinstance(value, np.ndarray):
                settings_dict[key] = value.tolist()  # Convert np.ndarray to list
        
        return settings_dict

    def save_to_file(self, file_path):
        class NumpyArrayEncoder(JSONEncoder):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return JSONEncoder.default(self, o)
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.__dict__, file, indent=4, cls=NumpyArrayEncoder)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, validate_settings: bool = True) -> "SettingsMS2Deepscore":
        return cls(validate_settings=validate_settings, **d)

    @classmethod
    def from_json(cls, s: str, *, validate_settings: bool = True) -> "SettingsMS2Deepscore":
        return cls.from_dict(json.loads(s), validate_settings=validate_settings)


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
            # Coerce incoming values against defaults for consistency
            settings = _coerce_settings_dict(settings, self)
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown setting: {key}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SettingsEmbeddingEvaluator":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "SettingsEmbeddingEvaluator":
        return cls.from_dict(json.loads(s))

    def get_dict(self):
        """returns a dictionary representation of the settings"""
        return self.__dict__
