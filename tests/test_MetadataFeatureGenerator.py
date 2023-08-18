import pytest
from matchms import Metadata
from ms2deepscore.MetadataFeatureGenerator import (CategoricalToBinary,
                                                   MetadataFeatureGenerator,
                                                   OneHotEncoder,
                                                   StandardScaler)


@pytest.fixture
def metadata():
    return Metadata({"ionmode": "Negative"})


def test_metadatafeaturegenerator_not_implemented(metadata):
    gen = MetadataFeatureGenerator()
    
    with pytest.raises(NotImplementedError):
        gen.generate_features(metadata)
    
    with pytest.raises(NotImplementedError):
        MetadataFeatureGenerator.load_from_dict({})


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

    with pytest.raises(AssertionError):
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

    with pytest.raises(AssertionError):
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
