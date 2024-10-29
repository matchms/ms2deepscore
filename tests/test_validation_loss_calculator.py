import numpy as np
import pytest
from matchms.Spectrum import Spectrum
from ms2deepscore.models.loss_functions import LOSS_FUNCTIONS
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import (
    ValidationLossCalculator)
from tests.create_test_spectra import (pesticides_test_spectra,
                                       siamese_spectral_model)


@pytest.fixture()
def simple_test_spectra():
    """Creates many random versions of two very differntly looking types of spectra.
    They come with very different compound annotations so that a model should easily be able to learn those.
    """
    spectra = []
    for i in range(10):
        spectra.append(Spectrum(mz=np.sort(np.random.uniform(0, 100, 10)),
                                intensities=np.random.uniform(0.2, 1, 10),
                                metadata={
                                    "precursor_mz": i,
                                    "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                                    "inchi": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
                                    "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                                },
                                ))
        spectra.append(Spectrum(mz=np.sort(np.random.uniform(100, 200, 10)),
                                intensities=np.random.uniform(0.2, 1, 10),
                                metadata={"precursor_mz": i+10,
                                          "smiles": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@@H](C(=O)O)N)OC(=O)CCCCCCCCCCCCCCCCC",
                                          "inchi": "InChI=1S/C42H82NO10P/c1-3-5-7-9-11-13-15-17-19-21-23-25-27-29-31-33-40(44)50-35-38(36-51-54(48,49)52-37-39(43)42(46)47)53-41(45)34-32-30-28-26-24-22-20-18-16-14-12-10-8-6-4-2/h38-39H,3-37,43H2,1-2H3,(H,46,47)(H,48,49)/t38-,39+/m1/s1",
                                          "inchikey": "TZCPCKNHXULUIY-RGULYWFUSA-N", },
                                ))
    return spectra


def test_validation_loss_calculator():
    model = siamese_spectral_model()
    test_spectra = pesticides_test_spectra()
    bins = np.array([(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
                     (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (-0.01, 0.1)])
    validation_loss_calculator = ValidationLossCalculator(test_spectra,
                                                          settings=SettingsMS2Deepscore(same_prob_bins=bins))

    losses, binned_losses = validation_loss_calculator.compute_binned_validation_loss(model,
                                                                       LOSS_FUNCTIONS.keys())
    assert len(losses) == len(LOSS_FUNCTIONS)
    assert len(binned_losses) == len(LOSS_FUNCTIONS)
