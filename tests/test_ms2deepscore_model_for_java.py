from math import isclose

import numpy as np
import torch
from matchms import Spectrum

from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model


def test_MS2DeepScore_score_pair():
    """Test score calculation using *.pair* method."""
    model = load_model(
        "../../data/pytorch/gnps_21_08_23_min_5_at_5_percent/trained_models/both_mode_precursor_mz_ionmode_2000_2000_2000_layers_500_embedding_2024_01_31_11_51_10/ms2deepscore_model.pt")
    similarity_measure = MS2DeepScore(model)

    test_spectrum_1 = Spectrum(mz=np.array([1., 2., 3.]),
                               intensities=np.array([0.1, 0.1, 0.1]),
                               metadata={"precursor_mz": 100,
                                         "ionmode": "positive"}
                               )
    test_spectrum_2 = Spectrum(mz=np.array([10., 20., 30.]),
                               intensities=np.array([0.1, 0.1, 0.1]),
                               metadata={"precursor_mz": 100,
                                         "ionmode": "positive"}
                               )

    score = similarity_measure.pair(test_spectrum_1, test_spectrum_2)
    print(score)
    # assert np.allclose(score, 0.990366, atol=1e-6), "Expected different score."
    # assert isinstance(score, float), "Expected score to be float"

def create_test_tensors():
    bin_size = 9900
    metadata_size = 2
    spectrum_1_value = 0.1
    torch.tensor([np.array([spectrum_1_value] * bin_size)], dtype=torch.float32),
    torch.tensor([np.array([1.] * bin_size)], dtype=torch.float32),
    torch.tensor([np.array([0.] * metadata_size)], dtype=torch.float32),
    torch.tensor([np.array([1.] * metadata_size)], dtype=torch.float32)


def test_siamese_model_forward_pass():
    model = load_model("../../../ms2deepscore/ms2deepscore/tests/resources/ms2deepscore_model.pt")
    similarity_score = model(torch.tensor([np.array([0.1]*990), np.array([0.2]*990)], dtype=torch.float32),
                             torch.tensor([np.array([0.2]*990), np.array([0.1]*990)], dtype=torch.float32),
                             torch.tensor([np.array([0.] * 2), np.array([1.] * 2)], dtype=torch.float32),
                             torch.tensor([np.array([1.] * 2), np.array([0.] * 2)], dtype=torch.float32))
    assert similarity_score.shape[0] == 2
    print(similarity_score)


def test_siamese_model_embedding_generation_from_tensor():
    """This test is to compare output of a test model with the output in MZMine for the same model"""
    model = load_model("../../../ms2deepscore/ms2deepscore/tests/resources/ms2deepscore_model.pt")
    similarity_score = model.encoder(torch.tensor([np.array([0.1]*990), np.array([0.2]*990)], dtype=torch.float32),
                  torch.tensor([np.array([0.] * 2), np.array([1.] * 2)], dtype=torch.float32),
                  )
    assert similarity_score.shape == (2, 50)
    assert isclose(float(similarity_score[0][0]), -4.6007e-02, abs_tol=0.001)
    assert isclose(float(similarity_score[1][0]), -3.7386e-02, abs_tol=0.001)


def test_siamese_model_embedding_generation_from_spectrum():
    model = load_model("../../../ms2deepscore/ms2deepscore/tests/resources/ms2deepscore_model.pt")
    ms2deepscore_model = MS2DeepScore(model)

    test_spectra = [Spectrum(mz=np.array([100.1, 200.1, 300.1, 400.1, 500.1]), intensities=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
                             metadata={"precursor_mz": 600,
                                       "ionmode": "positive"
                                       }),
                    Spectrum(mz=np.array([600.1, 700.1, 800.1, 900.1, 1000.1]), intensities=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
                             metadata={"precursor_mz": 1000,
                                       "ionmode": "positive"
                                       })]

    embeddings = ms2deepscore_model.get_embedding_array(test_spectra)
    print(embeddings)
    assert embeddings.shape == (2, 50)
    scores = ms2deepscore_model.matrix(test_spectra, test_spectra)
    print(scores)

