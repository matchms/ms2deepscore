import json
from importlib import import_module
from typing import List, Tuple, Union
import torch
from matchms import Metadata
from matchms.typing import SpectrumType
from tqdm import tqdm


class MetadataVectorizer:
    """Create a numerical vector of selected metadata fields including transformations.

    This class transforms a list of spectra into numerical vectors
    based on the specified metadata fields. These vectors can be used for 
    further analysis or machine learning models.

    Attributes
    ----------
    additional_metadata : list of MetadataFeatureGenerator
        List of metadata feature generators used to create the metadata vector.
    """

    def __init__(self, 
                 additional_metadata=()):
        """
        Parameters
        ----------
        additional_metadata : list of MetadataFeatureGenerator, optional
            List of metadata feature generators used to create the metadata vector. 
            Each element in the list should be an instance of a class that inherits from 
            MetadataFeatureGenerator. Default is an empty tuple.
        """
        self.additional_metadata = additional_metadata

    def transform(self, spectra: List[SpectrumType],
                  progress_bar=False):
        """Transforms the input spectra into metadata vectors as needed for MS2DeepScore.

        This method converts a list of spectra into numerical vectors based on the 
        specified metadata fields. Each spectrum's metadata is processed through 
        the feature generators specified in `additional_metadata`.

        Parameters
        ----------
        spectra : list of SpectrumType
            List of spectra to be transformed.
        progress_bar : bool, optional
            Show progress bar if set to True. Default is False..

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
        """Returns the size of the metadata vector.

        The size is determined by the number of feature generators specified 
        in `additional_metadata`.
        """
        return len(self.additional_metadata)


class MetadataFeatureGenerator:
    """Base class to define metadata-to-feature conversion rules.

    This class should be inherited by specific feature generator implementations
    that define how to convert metadata fields into numerical features.
    """
    def generate_features(self, metadata: Metadata) -> float:
        """This method should be implemented by child classes to generate a input feature for the model"""
        raise NotImplementedError

    def to_json(self) -> str:
        """Convert the feature generator to a JSON string.
        """
        return json.dumps((type(self).__name__, vars(self)))

    @classmethod
    def load_from_dict(cls, json_dict: dict):
        """This method should be implemented by child classes Create class instance from json.
        """
        raise NotImplementedError

    def __eq__(self, other):
        """Check equality with another feature generator.

        Parameters
        ----------
        other : MetadataFeatureGenerator
            The other feature generator to compare with.
        """
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__
        return False


class StandardScaler(MetadataFeatureGenerator):
    """StandardScaler scales metadata by subtracting the mean and dividing by the standard deviation.
    """
    def __init__(self, metadata_field: str, mean: float, std: float = None):
        """
        Parameters
        ----------
        metadata_field : str
            The metadata field to be scaled.
        mean : float
            The mean value used for scaling.
        standard_deviation : float, optional
            The standard deviation used for scaling. If not provided, only the mean will be subtracted.
        """
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
    """OneHotEncoder converts categorical metadata to binary (0 or 1) based on specified entries.
    """
    def __init__(self, metadata_field: str,
                 entries_becoming_one):
        """
        Parameters
        ----------
        metadata_field : str
            The metadata field to be encoded.
        entries_becoming_one : list or str or int
            The entries in the metadata field that should be encoded as 1.
        """
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
    """Converts categorical features (e.g., strings) into binary 1 or 0 feature values.

    Attributes
    ----------
    metadata_field : str
        The metadata field to be converted.
    entries_becoming_one : list or str or int
        The entries in the metadata field that should be encoded as 1.
    entries_becoming_zero : list or str or int
        The entries in the metadata field that should be encoded as 0.
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
    """Creates an object from json for any of the subclasses of MetadataFeatureGenerator.

    This function is used for loading instances of MetadataFeatureGenerator subclasses 
    from their JSON representations. Each JSON representation should include the class 
    name and the settings needed to initialize an instance of that class.

    Parameters
    ----------
    list_of_json_metadata_feature_generators : list of tuples
        A list containing tuples where each tuple consists of a class name (str) and a 
        dictionary of settings (dict) representing the JSON configuration of a 
        MetadataFeatureGenerator subclass.

    Returns
    -------
    tuple
        A tuple containing instances of the MetadataFeatureGenerator subclasses created 
        from the JSON configurations.

    Raises
    ------
    TypeError
        If any of the class names do not correspond to a subclass of MetadataFeatureGenerator.

    Example code:

    .. code-block:: python
        json_config = [
            ("StandardScaler", {"metadata_field": "intensity", "mean": 100.0, "standard_deviation": 15.0}),
            ("OneHotEncoder", {"metadata_field": "instrument_type", "entries_becoming_one": "FTICR"})
        ]
        feature_generators = load_from_json(json_config)
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
