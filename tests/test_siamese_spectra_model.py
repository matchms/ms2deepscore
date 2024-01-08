import pytest
import numpy as np
from matchms import Spectrum
from ms2deepscore.models import SiameseSpectralModel
from ms2deepscore.MetadataFeatureGenerator import (MetadataVectorizer,
                                                   StandardScaler)
from ms2deepscore.data_generators import tensorize_spectra


@pytest.fixture
def dummy_spectra():
    spectrum1 = Spectrum(mz=np.array([101, 202, 303.0]),
                         intensities=np.array([0.1, 0.2, 1.0]),
                         metadata={"precursor_mz": 222.2}
                         )
    spectrum2 = Spectrum(mz=np.array([101.5, 202.5, 303.0]),
                         intensities=np.array([0.1, 0.2, 1.0]),
                         metadata={"precursor_mz": 333.3})
    return [spectrum1, spectrum2]


def test_siamese_model_defaults():
    # Create the model instance
    model = SiameseSpectralModel(peak_inputs=9900, additional_inputs=0)

    assert model.model_parameters == {
        'base_dims': (1000, 800, 800),
        'embedding_dim': 400,
        'train_binning_layer': True,
        'group_size': 30,
        'output_per_group': 3,
        'dropout_rate': 0.2,
        'peak_inputs': 9900,
        'additional_inputs': 0
    }


def test_siamese_model_forward_pass(dummy_spectra):
    model = SiameseSpectralModel(peak_inputs=990, additional_inputs=0)
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, None, 10, 1000, 1, 0.5)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert similarity_score.shape[0] == 2


def test_siamese_model_no_binning_layer(dummy_spectra):
    model = SiameseSpectralModel(peak_inputs=990, additional_inputs=0, train_binning_layer=False)
    assert not model.model_parameters["train_binning_layer"]

    # Test forward pass
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, None, 10, 1000, 1, 0.5)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert similarity_score.shape[0] == 2


def test_siamese_model_additional_metadata(dummy_spectra):
    scaler = StandardScaler("precursor_mz", 200.0, 250.0)
    vectorizer = MetadataVectorizer([scaler])
    model = SiameseSpectralModel(peak_inputs=9900, additional_inputs=1, train_binning_layer=False)

    # Test forward pass
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, vectorizer, 10, 1000, 0.1, 0.5)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert similarity_score.shape[0] == 2
    assert model.encoder.dense_layers[0].weight.shape[1] == 9901

    # Include dense binning layer
    model = SiameseSpectralModel(peak_inputs=9900, additional_inputs=1)

    # Test forward pass
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, vectorizer, 10, 1000, 0.1, 0.5)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert model.encoder.dense_layers[0].weight.shape[1] == 991

    # Compare to no metadata_vectorizer
    model = SiameseSpectralModel(peak_inputs=9900, additional_inputs=0)

    # Test forward pass
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, None, 10, 1000, 0.1, 0.5)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert model.encoder.dense_layers[0].weight.shape[1] == 990
