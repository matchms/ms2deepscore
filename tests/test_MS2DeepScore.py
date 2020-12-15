import pytest
from ms2deepscore import MS2DeepScore


def test_MS2DeepScore():
    """Test if default initalization works"""
    ms2ds_model = MS2DeepScore(100)
    assert ms2ds_model.mz_max == 1000.0, "Expected different default value."
    assert ms2ds_model.mz_min == 10.0, "Expected different default value."
    assert ms2ds_model.d_bins == 9.9, "Expected differnt calculated bin size."