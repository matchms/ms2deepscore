import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore import SpectrumBinner
from ms2deepscore.MetadataFeatureGenerator import (CategoricalToBinary,
                                                   StandardScaler)


def test_SpectrumBinner():
    """Test if default initalization works"""
    ms2ds_binner = SpectrumBinner(100)
    assert ms2ds_binner.mz_max == 1000.0, "Expected different default value."
    assert ms2ds_binner.mz_min == 10.0, "Expected different default value."
    assert ms2ds_binner.d_bins == 9.9, "Expected different calculated bin size."


def test_SpectrumBinner_set_min_max():
    """Test if other limits work well"""
    ms2ds_binner = SpectrumBinner(100, mz_min=0.0, mz_max=100.0)
    assert ms2ds_binner.mz_max == 100.0, "Expected different default value."
    assert ms2ds_binner.mz_min == 0.0, "Expected different default value."
    assert ms2ds_binner.d_bins == 1.0, "Expected different calculated bin size."


def test_SpectrumBinner_fit_transform():
    """Test if collect binned spectrums method works."""
    ms2ds_binner = SpectrumBinner(100, mz_min=0.0, mz_max=100.0, peak_scaling=1.0)
    spectrum_1 = Spectrum(mz=np.array([10, 50, 100.]),
                          intensities=np.array([0.7, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 40, 90.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 40, 50, 90, 100], "Expected different known bins."
    assert len(binned_spectrums) == 2, "Expected 2 binned spectrums."
    assert binned_spectrums[0].binned_peaks == {0: 0.7, 2: 0.2, 4: 0.1}, \
        "Expected different binned spectrum."
    assert binned_spectrums[0].get("inchikey") == "test_inchikey_01", \
        "Expected different inchikeys."


def test_SpectrumBinner_fit_transform_peak_overlap():
    """Test if method works and takes the maximum peak intensity per bin."""
    ms2ds_binner = SpectrumBinner(100, mz_min=0.0, mz_max=100.0, peak_scaling=1.0)
    spectrum_1 = Spectrum(mz=np.array([10, 10.01, 100.]),
                          intensities=np.array([0.1, 0.8, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 40, 90.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 40, 90, 100], "Expected different known bins."
    assert binned_spectrums[0].binned_peaks == {0: 0.8, 3: 0.1}, \
        "Expected different binned spectrum."


def test_SpectrumBinner_fit_transform_peak_scaling():
    """Test if collect binned spectrums method works with different peak_scaling."""
    ms2ds_binner = SpectrumBinner(100, mz_min=0.0, mz_max=100.0, peak_scaling=0.0)
    spectrum_1 = Spectrum(mz=np.array([10, 50, 100.]),
                          intensities=np.array([0.7, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 40, 90.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 40, 50, 90, 100], "Expected different known bins."
    assert len(binned_spectrums) == 2, "Expected 2 binned spectrums."
    assert binned_spectrums[0].binned_peaks == {0: 1.0, 2: 1.0, 4: 1.0}, \
        "Expected different binned spectrum."
    assert binned_spectrums[0].get("inchikey") == "test_inchikey_01", \
        "Expected different inchikeys."


def test_SpectrumBinner_transform():
    """Test if creating binned spectrums method works."""
    ms2ds_binner = SpectrumBinner(100, mz_min=0.0, mz_max=100.0, peak_scaling=1.0)
    spectrum_1 = Spectrum(mz=np.array([10, 20, 50, 100.]),
                          intensities=np.array([0.7, 0.6, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 30, 40, 90.]),
                          intensities=np.array([0.4, 0.5, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 20, 30, 40, 50, 90, 100], "Expected different known bins."

    spectrum_3 = Spectrum(mz=np.array([10, 20, 30, 50.]),
                      intensities=np.array([0.4, 0.5, 0.2, 1.0]),
                      metadata={'inchikey': "test_inchikey_03"})
    spectrum_binned = ms2ds_binner.transform([spectrum_3])
    assert spectrum_binned[0].binned_peaks == {0: 0.4, 1: 0.5, 2: 0.2, 4: 1.0}, \
        "Expected different binned spectrum"


def test_SpectrumBinner_transform_missing_fraction():
    """Test if creating binned spectrums method works if peaks are unknown."""
    ms2ds_binner = SpectrumBinner(100, mz_min=0.0, mz_max=100.0, peak_scaling=1.0)
    spectrum_1 = Spectrum(mz=np.array([10, 20, 50, 100.]),
                          intensities=np.array([0.7, 0.6, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 30, 40, 90.]),
                          intensities=np.array([0.4, 0.5, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 20, 30, 40, 50, 90, 100], "Expected different known bins."

    spectrum_3 = Spectrum(mz=np.array([10, 20, 30, 80.]),
                      intensities=np.array([0.4, 0.5, 0.2, 1.0]),
                      metadata={'inchikey': "test_inchikey_03"})
    with pytest.raises(AssertionError) as msg:
        _ = ms2ds_binner.transform([spectrum_3])
    assert "weighted spectrum is unknown to the model"in str(msg.value), \
        "Expected different exception."


def test_spectrum_binner_additional_metadata():
    ms2ds_binner = SpectrumBinner(100, mz_min=0.0, mz_max=100.0, peak_scaling=1.0,
                                  additional_metadata=(StandardScaler("precursor_mz", mean=0, std=1000),
                                                       CategoricalToBinary("ionization_mode", "positive", "negative")))
    
    spectrum_1 = Spectrum(mz=np.array([10, 20, 50, 100.]),
                          intensities=np.array([0.7, 0.6, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01", "parent_mass": "100", "precursor_mz": "99",
                                    "ionization_mode": "positive"})
    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1])
    assert len(binned_spectrums[0].metadata) == 3, "Expected 3 items but found " + str(len(binned_spectrums[0].metadata)) + " in metadata."
    
    spectrum_2 = Spectrum(mz=np.array([10, 20, 50, 100.]),
                          intensities=np.array([0.7, 0.6, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01", "parent_mass": "100", "ionization_mode": "negative"})
    with pytest.raises(AssertionError) as msg:
        _= ms2ds_binner.transform([spectrum_2])
