from matchms import Metadata
from typing import Tuple


class MetadataFeatureGenerator:
    """Base class to define metadata-to-feature conversion rules.
    """
    def __init__(self, metadata: Metadata):
        self.metadata = metadata

    def generate_features(self) -> float:
        """This method should be implemented by child classes to generate a input feature for the model"""
        raise NotImplementedError


class PrecursorMZFeatureGenerator(MetadataFeatureGenerator):
    """Class for generating feature from precursor-m/z (here: simply m/z divided by 1000).
    """
    def generate_features(self) -> float:
        precursor_mz = self.metadata.get("precursor_mz")
        assert precursor_mz is not None, "No precursor mz was found, preprocess your spectra first using matchms"
        return precursor_mz/1000


class IonizationModeFeatureGenerator(MetadataFeatureGenerator):
    """Class for generating feature from ionization mode. Here simply 0: positive mode and 1: negative mode.
    """
    def generate_features(self) -> float:
        ionization_mode = self.metadata.get("ionization_mode")
        if ionization_mode == "positive":
            return 0
        if ionization_mode == "negative":
            return 1
        assert False, "Ionization mode should be 'positive' or 'negative'"
