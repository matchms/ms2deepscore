from collections import Counter

import numpy as np
import pytest
from matchms import Spectrum


from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    compute_jaccard_similarity_per_bin, convert_pair_array_to_coo_array,
    SelectedInchikeyPairs,
    select_inchi_for_unique_inchikeys, select_compound_pairs_wrapper, compute_fingerprints_for_training,
    convert_selected_pairs_matrix)
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
def test_spectra():
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


def test_select_inchi_for_unique_inchikeys(test_spectra):
    test_spectra[2].set("inchikey", "ABCABCABCABCAB-nonsense2")
    test_spectra[3].set("inchikey", "ABCABCABCABCAB-nonsense3")
    (spectrums_selected, inchikey14s) = select_inchi_for_unique_inchikeys(test_spectra)
    assert inchikey14s == ['ABCABCABCABCAB']
    assert spectrums_selected[0].get("inchi").startswith("InChI=1/C6H8O6/")


def test_select_inchi_for_unique_inchikeys_two_inchikeys(test_spectra):
    # Test for two different inchikeys
    (spectrums_selected, inchikey14s) = select_inchi_for_unique_inchikeys(test_spectra)
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


def test_select_compound_pairs_wrapper_no_resampling():
    spectrums = create_test_spectra(num_of_unique_inchikeys=26, num_of_spectra_per_inchikey=2)
    bins = [(0.5, 0.75), (0.25, 0.5), (0.75, 1), (-0.000001, 0.25)]
    max_pair_resampling = 1
    settings = SettingsMS2Deepscore(same_prob_bins=np.array(bins),
                                    average_pairs_per_bin=5,
                                    batch_size=8,
                                    max_pair_resampling=max_pair_resampling)
    selected_inchikey_pairs = select_compound_pairs_wrapper(spectrums, settings)

    check_balanced_scores_selecting_inchikey_pairs(selected_inchikey_pairs, bins)
    check_balanced_inchikey_count_selecting_inchikey_pairs(selected_inchikey_pairs)
    print_balanced_bins_per_inchikey(selected_inchikey_pairs, settings, spectrums)
    check_correct_oversampling(selected_inchikey_pairs, max_pair_resampling)


def test_select_compound_pairs_wrapper_with_resampling():
    spectrums = create_test_spectra(num_of_unique_inchikeys=26, num_of_spectra_per_inchikey=1)
    bins = [(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
            (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (-0.01, 0.1)]
    max_pair_resampling = 10
    settings = SettingsMS2Deepscore(same_prob_bins=np.array(bins, dtype="float32"),
                                    average_pairs_per_bin=10,
                                    batch_size=8,
                                    max_pair_resampling=max_pair_resampling)
    selected_inchikey_pairs = select_compound_pairs_wrapper(spectrums, settings)

    check_balanced_scores_selecting_inchikey_pairs(selected_inchikey_pairs, bins)
    check_balanced_inchikey_count_selecting_inchikey_pairs(selected_inchikey_pairs)
    # Currently doesn't check anything, but prints badly distributed pairs and the available pairs. It is hard to write
    # a good test, since the balancing behaviour we would like to see only happens when you have a lot more pairs
    # (and inchikeys) which is not suitable for a test.
    print_balanced_bins_per_inchikey(selected_inchikey_pairs, settings, spectrums)
    check_correct_oversampling(selected_inchikey_pairs, max_pair_resampling)


def check_correct_oversampling(selected_inchikey_pairs: SelectedInchikeyPairs, max_resampling: int):
    pair_counts = Counter(selected_inchikey_pairs.selected_inchikey_pairs)
    for count in pair_counts.values():
        assert count <= max_resampling, "the resampling was done too frequently"


def get_available_score_distribution(settings, spectra):
    """Gets the score distribution for the available pairs (before doing balanced selection)"""
    fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
        spectra,
        settings.fingerprint_type,
        settings.fingerprint_nbits)

    available_pairs_per_bin_matrix, available_scores_per_bin_matrix = compute_jaccard_similarity_per_bin(
        fingerprints,
        settings.max_pairs_per_bin,
        settings.same_prob_bins,
        settings.include_diagonal)

    available_pairs_per_bin = convert_selected_pairs_matrix(available_pairs_per_bin_matrix,
                                                            available_scores_per_bin_matrix, inchikeys14_unique)

    score_distribution_per_inchikey = {inchikey: [0]*len(settings.same_prob_bins) for inchikey in inchikeys14_unique}
    for available_pairs in available_pairs_per_bin:
        for inchikey_1, inchikey_2, score in available_pairs:
            for i, score_bin in enumerate(settings.same_prob_bins):
                if score > score_bin[0] and score <= score_bin[1]:
                    score_distribution_per_inchikey[inchikey_1][i] += 1
                    score_distribution_per_inchikey[inchikey_2][i] += 1
    return score_distribution_per_inchikey


def print_balanced_bins_per_inchikey(selected_inchikey_pairs: SelectedInchikeyPairs, settings, spectra):
    """Prints the availabl distribution and the balanced distribution

    Currently doesn't do any checks, because it is hard to check if the wanted behaviour is achieved,
    since it is different for small test sets compared to large test sets."""
    score_distribution_per_inchikey = {inchikey: [0]*len(settings.same_prob_bins) for inchikey in selected_inchikey_pairs.get_inchikey_counts().keys()}
    for inchikey_1, inchikey_2, score in selected_inchikey_pairs.selected_inchikey_pairs:
        for i, score_bin in enumerate(settings.same_prob_bins):
            if score > score_bin[0] and score <= score_bin[1]:
                score_distribution_per_inchikey[inchikey_1][i] += 1
                score_distribution_per_inchikey[inchikey_2][i] += 1
    available_score_distribution = get_available_score_distribution(settings, spectra)

    for inchikey in score_distribution_per_inchikey.keys():
        balanced_distribution = score_distribution_per_inchikey[inchikey]
        average_balanced_distribution = sum(balanced_distribution)/len(balanced_distribution)
        if min(balanced_distribution)*2 < average_balanced_distribution:
            available_distribution = available_score_distribution[inchikey]
            index_of_min = balanced_distribution.index(min(balanced_distribution))
            print(available_distribution, balanced_distribution)
            # assert minimum_available_distribution*settings.max_pair_resampling == min(balanced_distribution)


def check_balanced_inchikey_count_selecting_inchikey_pairs(selected_inchikey_pairs: SelectedInchikeyPairs):
    """Test if SelectedInchikeyPairs has an equal inchikey distribution
    """
    inchikey_counts = selected_inchikey_pairs.get_inchikey_counts()
    max_difference_in_inchikey_freq = max(inchikey_counts.values()) - min(inchikey_counts.values())
    assert max_difference_in_inchikey_freq < max(inchikey_counts.values())/2, "The frequency of the sampling of the inchikeys is too different"


def check_balanced_scores_selecting_inchikey_pairs(selected_inchikey_pairs: SelectedInchikeyPairs,
                                                   score_bins):
    """Test if SelectedInchikeyPairs has an equal inchikey distribution
    """
    scores = selected_inchikey_pairs.get_scores()
    # converting to float32 is required, since the scores are float32, otherwise equal numbers are seen as not equal
    # and put in the wrong bin.
    score_bins = np.array(score_bins, dtype="float32")
    score_bin_counts = {tuple(score_bin): 0 for score_bin in score_bins}
    for score in scores:
        for min_bound, max_bound in score_bins:
            if score > min_bound and score <= max_bound:
                score_bin_counts[(min_bound, max_bound)] += 1
    # Check that the number of pairs per bin is equal for all bins
    assert len(set(score_bin_counts.values())) == 1

