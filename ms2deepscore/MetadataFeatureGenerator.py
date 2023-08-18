import json
from importlib import import_module
from typing import List, Union
from matchms import Metadata


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
        assert self.metadata_field is not None, f"Metadata entry for {self.metadata_field} is missing."
        assert isinstance(feature, (int, float))
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
        assert self.metadata_field is not None, f"Metadata entry for {self.metadata_field} is missing."
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
        assert self.metadata_field is not None, f"Metadata entry for {self.metadata_field} is missing."
        if feature in self.entries_becoming_one:
            return 1
        if feature in self.entries_becoming_zero:
            return 0
        assert False, f"Feature should be {self.entries_becoming_one} or {self.entries_becoming_zero}, not {feature}"

    @classmethod
    def load_from_dict(cls, json_dict: dict):
        """Create FeatureToBinary instance from json.
        """
        return cls(json_dict["metadata_field"],
                   json_dict["entries_becoming_one"],
                   json_dict["entries_becoming_zero"],)


def load_from_json(list_of_json_metadata_feature_generators: List[str]):
    """Creates an object from json for any of the subclasses of MetadataFeatureGenerator

    This is used for loading in the MetadataFeatureGenerator in SpectrumBinner.

    list_of_json_metadata_feature_generators:
        A list containing all the json representations of the subclasses of MetadataFeatureGenerator.
    """
    possible_metadata_classes = import_module(__name__)
    metadata_feature_generator_list = []
    for metadata_feature_json in list_of_json_metadata_feature_generators:
        metadata_feature = json.loads(metadata_feature_json)
        class_name, settings = metadata_feature
        # loads in all the classes in MetadataFeatureGenerator.py
        metadata_class = getattr(possible_metadata_classes, class_name)
        assert issubclass(metadata_class, MetadataFeatureGenerator), "Unknown feature generator class."
        metadata_feature_generator_list.append(metadata_class.load_from_dict(settings))
    return tuple(metadata_feature_generator_list)
