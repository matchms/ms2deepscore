import numpy as np
import pytest
from matchms import Spectrum


from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    compute_jaccard_similarity_per_bin, convert_pair_array_to_coo_array,
    SelectedInchikeyPairs,
    select_inchi_for_unique_inchikeys, select_compound_pairs_wrapper)
from tests.create_test_spectra import create_test_spectra


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


@pytest.fixture
def spectrums():
    metadata = {"precursor_mz": 101.1,
                "inchikey": "ABCABCABCABCAB-nonsense",
                "inchi": "InChI=1/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1"}
    spectrum_1 = Spectrum(mz=np.array([100.]),
                          intensities=np.array([0.7]),
                          metadata=metadata)
    spectrum_2 = Spectrum(mz=np.array([90.]),
                          intensities=np.array([0.4]),
                          metadata=metadata)
    spectrum_3 = Spectrum(mz=np.array([90.]),
                          intensities=np.array([0.4]),
                          metadata=metadata)
    spectrum_4 = Spectrum(mz=np.array([90.]),
                          intensities=np.array([0.4]),
                          metadata={"inchikey": 14 * "X",
                                    "inchi": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"})
    return [spectrum_1, spectrum_2, spectrum_3, spectrum_4]


@pytest.fixture
def dummy_spectrum_pairs():
    spectrum_pairs = [("Inchikey0", "Inchikey1", 0.8),
                      ("Inchikey0", "Inchikey2", 0.6),
                      ("Inchikey2", "Inchikey1", 0.3),
                      ("Inchikey2", "Inchikey2", 1.0)]
    return spectrum_pairs


def test_compute_jaccard_similarity_per_bin(simple_fingerprints):
    selected_pairs_per_bin, selected_scores_per_bin = compute_jaccard_similarity_per_bin(
        simple_fingerprints, max_pairs_per_bin=4)
    matrix_numba = convert_pair_array_to_coo_array(
        selected_pairs_per_bin, selected_scores_per_bin, simple_fingerprints.shape[0])
    
    # Uncompiled
    selected_pairs_per_bin, selected_scores_per_bin = compute_jaccard_similarity_per_bin.py_func(
        simple_fingerprints, max_pairs_per_bin=4)
    matrix_py = convert_pair_array_to_coo_array(
        selected_pairs_per_bin, selected_scores_per_bin, simple_fingerprints.shape[0])
    
    for matrix in [matrix_numba, matrix_py]:
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix.diagonal(), 1.0)
        assert matrix.nnz > 0  # Make sure there are some non-zero entries


def test_compute_jaccard_similarity_per_bin_exclude_diagonal(simple_fingerprints):
    selected_pairs_per_bin, selected_scores_per_bin = compute_jaccard_similarity_per_bin(
        simple_fingerprints, max_pairs_per_bin=4, include_diagonal=False)
    matrix_numba = convert_pair_array_to_coo_array(
        selected_pairs_per_bin, selected_scores_per_bin, simple_fingerprints.shape[0])

    # Uncompiled
    selected_pairs_per_bin, selected_scores_per_bin = compute_jaccard_similarity_per_bin.py_func(
        simple_fingerprints, max_pairs_per_bin=4, include_diagonal=False)
    matrix_py = convert_pair_array_to_coo_array(
        selected_pairs_per_bin, selected_scores_per_bin, simple_fingerprints.shape[0])

    for matrix in [matrix_numba, matrix_py]:
        diagonal = matrix.diagonal()
        assert np.all(diagonal == 0)  # Ensure no non-zero diagonal elements


def test_compute_jaccard_similarity_per_bin_correct_counts(fingerprints):
    selected_pairs_per_bin, selected_scores_per_bin = compute_jaccard_similarity_per_bin(
        fingerprints, max_pairs_per_bin=8)
    matrix_numba = convert_pair_array_to_coo_array(
        selected_pairs_per_bin, selected_scores_per_bin, fingerprints.shape[0])

    # Uncompiled
    selected_pairs_per_bin, selected_scores_per_bin = compute_jaccard_similarity_per_bin.py_func(
        fingerprints, max_pairs_per_bin=8)
    matrix_py = convert_pair_array_to_coo_array(
        selected_pairs_per_bin, selected_scores_per_bin, fingerprints.shape[0])

    expected_histogram = np.array([6,  8,  2, 10,  8, 6,  8,  8,  0,  8])
    for matrix in [matrix_numba, matrix_py]:
        dense_matrix = matrix.todense()
        matrix_histogram = np.histogram(dense_matrix, 10)
        assert np.all(matrix_histogram[0] == expected_histogram)


def test_select_inchi_for_unique_inchikeys(spectrums):
    spectrums[2].set("inchikey", "ABCABCABCABCAB-nonsense2")
    spectrums[3].set("inchikey", "ABCABCABCABCAB-nonsense3")
    (spectrums_selected, inchikey14s) = select_inchi_for_unique_inchikeys(spectrums)
    assert inchikey14s == ['ABCABCABCABCAB']
    assert spectrums_selected[0].get("inchi").startswith("InChI=1/C6H8O6/")


def test_select_inchi_for_unique_inchikeys_two_inchikeys(spectrums):
    # Test for two different inchikeys
    (spectrums_selected, inchikey14s) = select_inchi_for_unique_inchikeys(spectrums)
    assert inchikey14s == ['ABCABCABCABCAB', 'XXXXXXXXXXXXXX']
    assert [s.get("inchi")[:15] for s in spectrums_selected] == ['InChI=1/C6H8O6/', 'InChI=1S/C8H10N']


def test_SelectedInchikeyPairs_generator_with_shuffle(dummy_spectrum_pairs):
    selected_inchikey_pairs = SelectedInchikeyPairs(dummy_spectrum_pairs)
    rng = np.random.default_rng(0)
    gen = selected_inchikey_pairs.generator(True, rng)

    found_pairs = []
    # do one complete loop
    for i in range(len(selected_inchikey_pairs)):
        found_pairs.append(next(gen))

    assert len(found_pairs) == len(dummy_spectrum_pairs)
    assert sorted(found_pairs) == sorted(dummy_spectrum_pairs)

    found_pairs = []
    # do one complete loop
    for i in range(len(selected_inchikey_pairs)):
        found_pairs.append(next(gen))

    assert len(found_pairs) == len(dummy_spectrum_pairs)
    assert sorted(found_pairs) == sorted(dummy_spectrum_pairs)


def test_SelectedInchikeyPairs_generator_without_shuffle(dummy_spectrum_pairs):
    selected_inchikey_pairs = SelectedInchikeyPairs(dummy_spectrum_pairs)
    gen = selected_inchikey_pairs.generator(False, None)

    for _, expected_pair in enumerate(dummy_spectrum_pairs):
        assert expected_pair == next(gen)


@pytest.fixture
def dummy_selected_inchikey_pairs() -> SelectedInchikeyPairs:
    spectrums = create_test_spectra(num_of_unique_inchikeys=17, num_of_spectra_per_inchikey=2)
    settings = SettingsMS2Deepscore(same_prob_bins=np.array([(-0.000001, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]),
                                    average_pairs_per_bin=2,
                                    batch_size=8)
    return select_compound_pairs_wrapper(spectrums, settings)


def test_balanced_inchikey_count_selecting_inchikey_pairs(dummy_selected_inchikey_pairs):
    """Test if SelectedInchikeyPairs has an equal inchikey distribution
    """
    inchikey_counts = dummy_selected_inchikey_pairs.get_inchikey_counts()
    max_difference_in_inchikey_freq = max(inchikey_counts.values()) - min(inchikey_counts.values())
    assert max_difference_in_inchikey_freq <= 2, "The frequency of the sampling of the inchikeys is too different"


def test_balanced_scores_selecting_inchikey_pairs(dummy_selected_inchikey_pairs):
    """Test if SelectedInchikeyPairs has an equal inchikey distribution
    """
    scores = dummy_selected_inchikey_pairs.get_scores()
    score_bins = [(-0.000001, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    score_bin_counts = {score_bin: 0 for score_bin in score_bins}
    for score in scores:
        for min_bound, max_bound in score_bin_counts.keys():
            if score > min_bound and score <= max_bound:
                score_bin_counts[(min_bound, max_bound)] += 1
    # Check that the number of pairs per bin is equal for all bins
    assert len(set(score_bin_counts.values())) == 1


def test_no_repeating_of_pairs_when_selecting_inchikey_pairs(dummy_selected_inchikey_pairs):
    """Pairs are stored as inchikey1, inchikey2 score, this test checks that no pairs reappear,
    including the reverse pairs like inchikey2, inchikey1"""
    pairs = dummy_selected_inchikey_pairs.selected_inchikey_pairs
    for inchikey1_check, inchikey2_check, _ in pairs:
        count_of_pair = 0
        for inchikey1, inchikey2, _ in pairs:
            if inchikey1 == inchikey1_check and inchikey2 == inchikey2_check:
                count_of_pair += 1
            elif inchikey1 == inchikey2_check and inchikey2 == inchikey1_check:
                count_of_pair += 1
        assert count_of_pair == 1, "The pair occurs multiple times in the selected pairs (likely in reversed form)"
