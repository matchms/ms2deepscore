from matchms import Metadata
from typing import Tuple


class MetadataFeatureGenerator:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata

    def generate_features(self) -> float:
        """This method should be implemented by child classes to generate a input feature for the model"""
        raise NotImplementedError

    @staticmethod
    def feature_name() -> str:
        """This method should be implemented by child classes to generate the name of the feature"""
        raise NotImplementedError


class PrecursorMZFeatureGenerator(MetadataFeatureGenerator):
    def generate_features(self) -> float:
        precursor_mz = self.metadata.get("precursor_mz")
        assert precursor_mz is not None, "No precursor mz was found, preprocess your spectra first using matchms"
        return precursor_mz/1000

    @staticmethod
    def feature_name() -> str:
        return "precursor_mz"


class IonizationModeFeatureGenerator(MetadataFeatureGenerator):
    def generate_features(self) -> float:
        ionization_mode = self.metadata.get("ionization_mode")
        if ionization_mode == "positive":
            return 0
        if ionization_mode == "negative":
            return 1
        assert False, "Ionization mode should be 'positive' or 'negative'"

    @staticmethod
    def feature_name() -> str:
        return "ionization_mode"
