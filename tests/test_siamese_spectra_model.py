import pytest
import numpy as np
from matchms import Spectrum
from ms2deepscore.models import SiameseSpectralModel


@pytest.fixture
def dummy_spectra():
    # Example inputs
    spectrum1 = Spectrum(mz=np.array([101, 202, 303.0]), intensities=np.array([0.1, 0.2, 1.0]))
    spectrum2 = Spectrum(mz=np.array([101.5, 202.5, 303.0]), intensities=np.array([0.1, 0.2, 1.0]))
    return [spectrum1, spectrum2]


def test_siamese_model_defaults():
    # Create the model instance
    model = SiameseSpectralModel()

    assert model.model_parameters == {
        'min_mz': 0,
        'max_mz': 1000,
        'mz_bin_width': 0.01,
        'base_dims': (1000, 800, 800),
        'embedding_dim': 400,
        'train_binning_layer': True,
        'group_size': 30,
        'output_per_group': 3,
        'dropout_rate': 0.2
    }


def test_siamese_model_forward_pass(dummy_spectra):
    model = SiameseSpectralModel()
    similarity_score = model([dummy_spectra])
    assert similarity_score.shape[0] == 1

    similarity_score = model([dummy_spectra, dummy_spectra])
    assert similarity_score.shape[0] == 2


def test_siamese_model_no_binning_layer(dummy_spectra):
    model = SiameseSpectralModel(train_binning_layer=False)
    assert not model.model_parameters["train_binning_layer"]

    # Test forward pass
    similarity_score = model([dummy_spectra])
    assert similarity_score.shape[0] == 1
