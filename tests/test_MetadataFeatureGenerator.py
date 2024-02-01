import numpy as np
import pytest
import torch
from matchms import Metadata, Spectrum
from ms2deepscore.MetadataFeatureGenerator import (CategoricalToBinary,
                                                   MetadataFeatureGenerator,
                                                   MetadataVectorizer,
                                                   OneHotEncoder,
                                                   StandardScaler,
                                                   load_from_json)


@pytest.fixture
def metadata():
    return Metadata({"ionmode": "Negative"})


def test_metadatafeaturegenerator_not_implemented(metadata):
    gen = MetadataFeatureGenerator()
    
    with pytest.raises(NotImplementedError):
        gen.generate_features(metadata)
    
    with pytest.raises(NotImplementedError):
        MetadataFeatureGenerator.load_from_dict({})


def test_metadata_vectorizer(metadata):
    scaler = StandardScaler("mass", 200.0, 250.0)
    metadata = {"mass": 220.0}
    s1 = Spectrum(mz=np.array([100.]), intensities=np.array([1.0]), metadata=metadata)
    vectorizer = MetadataVectorizer([scaler])
    expected_value = (220 - 200) / 250
    assert vectorizer.transform([s1]) == torch.tensor([expected_value])
    assert (vectorizer.transform([s1, s1]) == torch.tensor([expected_value, expected_value])).all()
    assert vectorizer.size == 1


def test_standard_scaler_generate_features_with_std():
    scaler = StandardScaler("mass", 5.0, 1.0)
    metadata = Metadata({"mass": 7.0})

    assert scaler.generate_features(metadata) == 2.0


def test_standard_scaler_generate_features_without_std():
    scaler = StandardScaler("mass", 5.0)
    metadata = Metadata({"mass": 7.0})

    assert scaler.generate_features(metadata) == 2.0


def test_standard_scaler_generate_features_assert():
    scaler = StandardScaler("mass", 5.0)
    metadata = Metadata()

    with pytest.raises(TypeError):
        scaler.generate_features(metadata)


def test_one_hot_encoder():
    encoder = OneHotEncoder("category", "A")
    metadata_present = Metadata({"category": "A"})
    metadata_absent = Metadata({"category": "B"})

    assert encoder.generate_features(metadata_present) == 1
    assert encoder.generate_features(metadata_absent) == 0


def test_categorical_to_binary():
    converter = CategoricalToBinary("category", "A", "B")
    metadata_a = Metadata({"category": "A"})
    metadata_b = Metadata({"category": "B"})

    assert converter.generate_features(metadata_a) == 1
    assert converter.generate_features(metadata_b) == 0

    with pytest.raises(ValueError):
        converter.generate_features(Metadata({"category": "C"}))


def test_categorical_to_binary_lists():
    converter = CategoricalToBinary("category", ["A", "D"], ["B", "E"])
    metadata_a = Metadata({"category": "A"})
    metadata_d = Metadata({"category": "D"})
    metadata_b = Metadata({"category": "B"})
    metadata_e = Metadata({"category": "E"})

    assert converter.generate_features(metadata_a) == 1
    assert converter.generate_features(metadata_d) == 1
    assert converter.generate_features(metadata_b) == 0
    assert converter.generate_features(metadata_e) == 0


def test_equality():
    scaler1 = StandardScaler("mass", 5.0, 1.0)
    scaler2 = StandardScaler("mass", 5.0, 1.0)
    scaler3 = StandardScaler("mass", 4.0, 1.0)
    assert scaler1 == scaler2
    assert scaler1 != scaler3


def test_load_from_json():
    feature_generators = load_from_json([("StandardScaler", {"metadata_field": "precursor_mz",
                                        "mean": 200.0,
                                        "standard_deviation": 250.0}),
                    ("CategoricalToBinary", {"metadata_field": "ionmode",
                                             "entries_becoming_one": "positive",
                                             "entries_becoming_zero": "negative"}),
                    ])
    assert len(feature_generators) == 2
    assert isinstance(feature_generators[0], StandardScaler)
    assert isinstance(feature_generators[1], CategoricalToBinary)
