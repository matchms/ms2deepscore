from typing import Optional
import numpy as np


class SettingsMS2Deepscore:
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
        self.nbits: int = 2048
        if settings:
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown setting: {key}")
