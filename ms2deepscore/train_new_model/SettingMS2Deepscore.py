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

        # training settings
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.epochs = 150

        # Compound pairs selection settings
        self.average_pairs_per_bin = 20
        self.max_pairs_per_bin = 100
        self.tanimoto_bins: np.ndarray = np.array([(x / 10, x / 10 + 0.1) for x in range(0, 10)])
        self.include_diagonal: bool = True
        self.random_seed: Optional[int] = None

        # Tanimioto score setings
        self.fingerprint_type: str = "daylight"
        self.fingerprint_nbits: int = 2048
        if settings:
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown setting: {key}")
