import json
from importlib import import_module
from typing import List, Tuple, Union
import torch
from matchms import Metadata
from matchms.typing import SpectrumType
from tqdm import tqdm


class MetadataVectorizer:
    """Create a numerical vector of selected metadata field including transformations..
    """

    def __init__(self, 
                 additional_metadata=()):
        """

        Parameters
        ----------
        additional_metadata:
            List of all metadata used/wanted in a metadata vector. Default is ().
        """
        self.additional_metadata = additional_metadata

    def transform(self, spectra: List[SpectrumType],
                  progress_bar=False):
        """Transforms the input *spectrums* into metadata vectors as needed for
        MS2DeepScore.

        Parameters
        ----------
        spectra
            List of spectra.
        progress_bar
            Show progress bar if set to True. Default is False.

        Returns:
            List of metadata vectors.
        """
        metadata_vectors = torch.zeros((len(spectra), self.size))
        for i, spec in tqdm(enumerate(spectra),
                         desc="Create metadata vectors",
                         disable=(not progress_bar)):
            metadata_vectors[i, :] = \
                torch.tensor([feature_generator.generate_features(spec.metadata)
                    for feature_generator in self.additional_metadata])
        return metadata_vectors

    @property
    def size(self):
        return len(self.additional_metadata)


class MetadataFeatureGenerator:
    """Base class to define metadata-to-feature conversion rules.
    """
    def generate_features(self, metadata: Metadata) -> float:
        """This method should be implemented by child classes to generate a input feature for the model"""
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps((type(self).__name__, vars(self)))

    @classmethod
    def load_from_dict(cls, json_dict: dict):
        """This method should be implemented by child classes Create class instance from json.
        """
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__
        return False


class StandardScaler(MetadataFeatureGenerator):
    def __init__(self, metadata_field: str, mean: float, std: float = None):
        self.metadata_field = metadata_field
        self.mean = mean
        self.standard_deviation = std

    def generate_features(self, metadata: Metadata):
        feature = metadata.get(self.metadata_field, None)
        if self.metadata_field is None:
            raise ValueError(f"Metadata entry for {self.metadata_field} is missing.")
        if not isinstance(feature, (int, float)):
            raise TypeError(f"Expected float or int, got {feature}, for {self.metadata_field}")
        if self.standard_deviation:
            return (feature - self.mean) / self.standard_deviation
        return feature - self.mean

    @classmethod
    def load_from_dict(cls, json_dict: dict):
        """Create StandardScaler instance from json.
        """
        return cls(json_dict["metadata_field"],
                   json_dict["mean"],
                   json_dict["standard_deviation"],)


class OneHotEncoder(MetadataFeatureGenerator):
    def __init__(self, metadata_field: str,
                 entries_becoming_one):
        self.metadata_field = metadata_field
        self.entries_becoming_one = entries_becoming_one

    def generate_features(self, metadata: Metadata):
        feature = metadata.get(self.metadata_field, None)
        if self.metadata_field is None:
            raise ValueError(f"Metadata entry for {self.metadata_field} is missing.")
        if feature == self.entries_becoming_one:
            return 1
        return 0

    @classmethod
    def load_from_dict(cls, json_dict: dict):
        """Create OneHotEncoder instance from json.
        """
        return cls(json_dict["metadata_field"],
                   json_dict["entries_becoming_one"])


class CategoricalToBinary(MetadataFeatureGenerator):
    """Converts categorical features (e.g. strings) into binary 1 or 0 feature values.
    """
    def __init__(self, metadata_field: str,
                 entries_becoming_one: Union[list, str, int],
                 entries_becoming_zero: Union[list, str, int]):
        self.metadata_field = metadata_field
        if isinstance(entries_becoming_one, list):
            self.entries_becoming_one = entries_becoming_one
        else:
            self.entries_becoming_one = [entries_becoming_one]
        if isinstance(entries_becoming_zero, list):
            self.entries_becoming_zero = entries_becoming_zero
        else:
            self.entries_becoming_zero = [entries_becoming_zero]

    def generate_features(self, metadata: Metadata):
        feature = metadata.get(self.metadata_field, None)
        if self.metadata_field is None:
            raise ValueError(f"Metadata entry for {self.metadata_field} is missing.")
        if feature in self.entries_becoming_one:
            return 1
        if feature in self.entries_becoming_zero:
            return 0
        raise ValueError(f"Feature should be {self.entries_becoming_one} or {self.entries_becoming_zero}, not {feature}")

    @classmethod
    def load_from_dict(cls, json_dict: dict):
        """Create FeatureToBinary instance from json.
        """
        return cls(json_dict["metadata_field"],
                   json_dict["entries_becoming_one"],
                   json_dict["entries_becoming_zero"],)


def load_from_json(list_of_json_metadata_feature_generators: List[Tuple[str, dict]]):
    """Creates an object from json for any of the subclasses of MetadataFeatureGenerator

    This is used for loading in the MetadataFeatureGenerator in SpectrumBinner.

    list_of_json_metadata_feature_generators:
        A list containing all the json representations of the subclasses of MetadataFeatureGenerator.
    """
    possible_metadata_classes = import_module(__name__)
    metadata_feature_generator_list = []
    for class_name, settings in list_of_json_metadata_feature_generators:
        # loads in all the classes in MetadataFeatureGenerator.py
        metadata_class = getattr(possible_metadata_classes, class_name)
        if not issubclass(metadata_class, MetadataFeatureGenerator):
            raise TypeError("Unknown feature generator class.")
        metadata_feature_generator_list.append(metadata_class.load_from_dict(settings))
    return tuple(metadata_feature_generator_list)
