from ms2deepscore import BinnedSpectrum


def test_BinnedSpectrum():
    """basic test for BinnedSpectrum class"""
    peaks = {'20': 0.05,
             '21': 0.1,
             '25': 0.5,
             '30': 0.5,
             '100': 1.0}
    metadata = {"inchikey": "test"}
    binned_spectrum = BinnedSpectrum(binned_peaks=peaks,
                                     metadata=metadata)
    assert binned_spectrum.binned_peaks["20"] == 0.05
    assert list(binned_spectrum.binned_peaks.keys()) == ['20', '21', '25', '30', '100']
    assert binned_spectrum.get("inchikey") == "test"
    assert binned_spectrum.get("smiles") is None


def test_BinnedSpectrum_deepcopy():
    """Test if metadata is made a correct deepcopy"""
    peaks = {'20': 0.05,
             '21': 0.1}
    metadata = {"inchikey": "test"}
    binned_spectrum = BinnedSpectrum(binned_peaks=peaks,
                                     metadata=metadata)
    inchikey = binned_spectrum.get("inchikey")
    inchikey = inchikey + "wrong_addition"
    assert binned_spectrum.get("inchikey") == "test"
