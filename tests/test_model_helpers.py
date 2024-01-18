import numpy as np
from ms2deepscore.models.helper_functions import risk_aware_loss


def test_risk_aware_loss():
    assert risk_aware_loss(0.5, 0.5)  == 0
    assert risk_aware_loss(0.7, 0.5)  == risk_aware_loss(0.3, 0.5) == 0.1
    assert np.allclose(risk_aware_loss(0.4, 0.8), 0.32)
