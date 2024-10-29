import string
from pathlib import Path
import numpy as np
from matchms import Spectrum
from ms2deepscore.validation_loss_calculation.calculate_scores_for_validation import calculate_tanimoto_scores_unique_inchikey

TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def create_dummy_data(nr_of_spectra):
    """Create fake data to test generators.
    """
    # Create fake smiles
    spectrums = []
    for i in range(nr_of_spectra):
        smiles = "C" * (i+1)
        dummy_inchikey = f"{14 * string.ascii_uppercase[i]}-{10 * string.ascii_uppercase[i]}-N"
        spectrum = Spectrum(mz=np.array([100.0 + (i+1) * 25.0]), intensities=np.array([0.1]),
                            metadata={"inchikey": dummy_inchikey,
                                      "smiles": smiles})
        spectrums.append(spectrum)
    return spectrums


def test_calculate_tanimoto_scores_unique_inchikey():
    """Tests that only scores are calculated between unique inchikeys"""
    nr_of_test_spectra = 4
    spectrums = create_dummy_data(nr_of_test_spectra)
    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
        spectrums + spectrums,
        spectrums)
    assert tanimoto_scores.shape == (nr_of_test_spectra, nr_of_test_spectra)
