import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore.spectrum_pair_selection import (
    compute_jaccard_similarity_matrix_cherrypicking,
    jaccard_similarity_matrix_cherrypicking,
    select_inchi_for_unique_inchikey
    )


@pytest.fixture
def simple_fingerprints():
    return np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)


@pytest.fixture
def fingerprints(): 
    return np.array([
        [1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
    ], dtype=bool)


def test_basic_functionality(simple_fingerprints):
    matrix = jaccard_similarity_matrix_cherrypicking(simple_fingerprints, random_seed=42)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix.diagonal(), 1.0)
    assert matrix.nnz > 0  # Make sure there are some non-zero entries


def test_exclude_diagonal(simple_fingerprints):
    matrix = jaccard_similarity_matrix_cherrypicking(simple_fingerprints, include_diagonal=False, random_seed=42)
    diagonal = matrix.diagonal()
    assert np.all(diagonal == 0)  # Ensure no non-zero diagonal elements


def test_correct_counts(fingerprints):
    matrix = jaccard_similarity_matrix_cherrypicking(fingerprints)
    expected_histogram = np.array([6,  8,  2, 10,  8, 14,  0,  8,  0,  8])
    assert np.all(np.histogram(matrix.todense(), 10)[0] == expected_histogram)


def test_global_bias(fingerprints):
    bins = np.array([(0, 0.5), (0.5, 0.8), (0.8, 1.0)])
    data, _, _ = compute_jaccard_similarity_matrix_cherrypicking(fingerprints,
                                                                 selection_bins=bins,
                                                                 max_pairs_per_bin=1)
    data = np.array(data)
    assert (data <= 0.5).sum() == ((data>0.5) & (data<=0.8)).sum() == (data>0.8).sum() == 8


def test_global_bias_not_possible(fingerprints):
    bins = np.array([(0, 0.5), (0.5, 0.8), (0.8, 1.0)])
    # Test uncompiled function
    data, _, _ = compute_jaccard_similarity_matrix_cherrypicking.py_func(
        fingerprints,
        selection_bins=bins,
        max_pairs_per_bin=2)
    data = np.array(data)
    assert (data <= 0.5).sum() == ((data>0.5) & (data<=0.8)).sum() == 16
    assert (data>0.8).sum() == 8

    # Test compiled function
    data, _, _ = compute_jaccard_similarity_matrix_cherrypicking(
        fingerprints,
        selection_bins=bins,
        max_pairs_per_bin=2)
    data = np.array(data)
    assert (data <= 0.5).sum() == ((data>0.5) & (data<=0.8)).sum() == 16
    assert (data>0.8).sum() == 8


def test_select_inchi_for_unique_inchikey():
    #ms2ds_binner = SpectrumBinner(100, mz_min=0.0, mz_max=100.0, peak_scaling=1.0)
    spectrum_1 = Spectrum(mz=np.array([100.]),
                          intensities=np.array([0.7]),
                          metadata={"inchikey": "ABCABCABCABCAB-nonsense",
                                    "inchi": "InChI=1/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1"})
    spectrum_2 = Spectrum(mz=np.array([90.]),
                          intensities=np.array([0.4]),
                          metadata={"inchikey": "ABCABCABCABCAB-nonsense",
                                    "inchi": "InChI=1/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1"})
    spectrum_3 = Spectrum(mz=np.array([90.]),
                          intensities=np.array([0.4]),
                          metadata={"inchikey": "ABCABCABCABCAB-nonsense2",
                                    "inchi": "InChI=1/C666H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1"})

    select_inchi_for_unique_inchikey([spectrum_1, spectrum_2, spectrum_3])
    assert ms2ds_binner.known_bins == [10, 40, 50, 90, 100], "Expected different known bins."
    assert len(binned_spectrums) == 2, "Expected 2 binned spectrums."
    assert binned_spectrums[0].binned_peaks == {0: 0.7, 2: 0.2, 4: 0.1}, \
        "Expected different binned spectrum."
    assert binned_spectrums[0].get("inchikey") == "test_inchikey_01", \
        "Expected different inchikeys."