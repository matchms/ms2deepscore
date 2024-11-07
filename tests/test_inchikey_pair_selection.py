from collections import Counter

import numpy as np
import pytest
from matchms import Spectrum


from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    compute_jaccard_similarity_per_bin, select_inchi_for_unique_inchikeys, select_compound_pairs_wrapper, compute_fingerprints_for_training)
from ms2deepscore.train_new_model import InchikeyPairGenerator
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
    max_pairs_per_bin = 5
    nr_of_bins = 10
    selection_bins = np.array([(x / nr_of_bins, x / nr_of_bins + 1/ nr_of_bins) for x in range(nr_of_bins)])
    selected_pairs_per_bin_numba, selected_scores_per_bin_numba = compute_jaccard_similarity_per_bin(
        simple_fingerprints, max_pairs_per_bin=max_pairs_per_bin,
        selection_bins=selection_bins)

    # Uncompiled
    selected_pairs_per_bin_py, selected_scores_per_bin_py = compute_jaccard_similarity_per_bin.py_func(
        simple_fingerprints, max_pairs_per_bin=max_pairs_per_bin,
        selection_bins=selection_bins)

    def check_correct_matrixes(selected_pairs_per_bin, selected_scores_per_bin):
        assert selected_pairs_per_bin.shape == (nr_of_bins, len(simple_fingerprints), max_pairs_per_bin)
        expected_nr_of_pairs_per_bin = np.array([0, 0, 4, 4, 0, 0, 2, 0, 0, 4])

        for bin_id, pairs_matrix in enumerate(selected_pairs_per_bin):

            number_of_pairs_in_bin = len(np.where(pairs_matrix != -1)[0])
            assert expected_nr_of_pairs_per_bin[bin_id] == number_of_pairs_in_bin

            assert np.all(selected_scores_per_bin[bin_id][pairs_matrix == -1] == 0), \
                "If no pair is available the score should be 0"
            assert np.all(selected_scores_per_bin[bin_id][pairs_matrix != -1] > 0.0), \
                "If a pair is found the score should not be 0 (in principle it could be, but not the case for these fingerprints)"

            if selection_bins[bin_id][1] == 1:
                for inchikey_idx, row in enumerate(pairs_matrix):
                    assert len(np.where(row == inchikey_idx)[0]) == 1, \
                        "When select_diagonal is True there should be a pair with itself in the bin including 1.0"
                    assert selected_scores_per_bin[bin_id][inchikey_idx][row == inchikey_idx] == 1.0
            else:
                for inchikey_idx, row in enumerate(pairs_matrix):
                    assert len(np.where(row == inchikey_idx)[0]) == 0, \
                        "The bins not including 1.0, should not have pairs between the same inchikey"

    check_correct_matrixes(selected_pairs_per_bin_numba, selected_scores_per_bin_numba)
    check_correct_matrixes(selected_pairs_per_bin_py, selected_scores_per_bin_py)


def test_compute_jaccard_similarity_per_bin_exclude_diagonal(simple_fingerprints):
    max_pairs_per_bin = 5
    nr_of_bins = 10
    selection_bins = np.array([(x / nr_of_bins, x / nr_of_bins + 1 / nr_of_bins) for x in range(nr_of_bins)])
    selected_pairs_per_bin_numba, selected_scores_per_bin_numba = compute_jaccard_similarity_per_bin(
        simple_fingerprints, max_pairs_per_bin=max_pairs_per_bin,
        selection_bins=selection_bins, include_diagonal=False)

    # Uncompiled
    selected_pairs_per_bin_py, selected_scores_per_bin_py = compute_jaccard_similarity_per_bin.py_func(
        simple_fingerprints, max_pairs_per_bin=max_pairs_per_bin,
        selection_bins=selection_bins, include_diagonal=False)

    def check_correct_matrixes(selected_pairs_per_bin):
        assert selected_pairs_per_bin.shape == (nr_of_bins, len(simple_fingerprints), max_pairs_per_bin)
        for bin_id, pairs_matrix in enumerate(selected_pairs_per_bin):
            for inchikey_idx, row in enumerate(pairs_matrix):
                assert len(np.where(row == inchikey_idx)[0]) == 0, \
                    "When include_diagonal is False there should not have pairs between the same inchikey"

    check_correct_matrixes(selected_pairs_per_bin_numba)
    check_correct_matrixes(selected_pairs_per_bin_py)


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
    selected_inchikey_pairs = InchikeyPairGenerator(dummy_spectrum_pairs)
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
    selected_inchikey_pairs = InchikeyPairGenerator(dummy_spectrum_pairs)
    gen = selected_inchikey_pairs.generator(False, None)

    for _, expected_pair in enumerate(dummy_spectrum_pairs):
        assert expected_pair == next(gen)


def test_select_compound_pairs_wrapper_no_resampling():
    spectrums = create_test_spectra(num_of_unique_inchikeys=26, num_of_spectra_per_inchikey=2)
    bins = [(0.5, 0.75), (0.25, 0.5), (0.75, 1), (-0.000001, 0.25)]
    max_pair_resampling = 1
    settings = SettingsMS2Deepscore(same_prob_bins=np.array(bins),
                                    average_inchikey_sampling_count=10,
                                    batch_size=8,
                                    max_pair_resampling=max_pair_resampling)
    selected_inchikey_pairs = select_compound_pairs_wrapper(spectrums, settings)
    inchikey_pair_generator = InchikeyPairGenerator(selected_inchikey_pairs)

    check_balanced_scores_selecting_inchikey_pairs(inchikey_pair_generator, bins)
    check_correct_oversampling(inchikey_pair_generator, max_pair_resampling)

    # Currently doesn't check anything, but prints badly distributed pairs and the available pairs. It is hard to write
    # a good test, since the balancing behaviour we would like to see only happens when you have a lot more pairs
    # (and inchikeys) which is not suitable for a test.
    print_balanced_bins_per_inchikey(inchikey_pair_generator, settings, spectrums)


def test_select_compound_pairs_wrapper_with_resampling():
    spectrums = create_test_spectra(num_of_unique_inchikeys=26, num_of_spectra_per_inchikey=1)
    bins = [(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
            (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (-0.01, 0.1)]
    max_pair_resampling = 10
    settings = SettingsMS2Deepscore(same_prob_bins=np.array(bins, dtype="float32"),
                                    average_inchikey_sampling_count=10,
                                    batch_size=8,
                                    max_pair_resampling=max_pair_resampling)
    selected_inchikey_pairs = select_compound_pairs_wrapper(spectrums, settings)
    inchikey_pair_generator = InchikeyPairGenerator(selected_inchikey_pairs)

    check_balanced_scores_selecting_inchikey_pairs(inchikey_pair_generator, bins)
    check_correct_oversampling(inchikey_pair_generator, max_pair_resampling)
    # Currently doesn't check anything, but prints badly distributed pairs and the available pairs. It is hard to write
    # a good test, since the balancing behaviour we would like to see only happens when you have a lot more pairs
    # (and inchikeys) which is not suitable for a test.
    print_balanced_bins_per_inchikey(inchikey_pair_generator, settings, spectrums)


def test_select_compound_pairs_wrapper_maximum_inchikey_count():
    spectrums = create_test_spectra(num_of_unique_inchikeys=26, num_of_spectra_per_inchikey=1)
    bins = [(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
            (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (-0.01, 0.1)]
    max_pair_resampling = 1000
    max_inchikey_sampling = 280
    settings = SettingsMS2Deepscore(same_prob_bins=np.array(bins, dtype="float32"),
                                    average_inchikey_sampling_count=200,
                                    batch_size=8,
                                    max_pair_resampling=max_pair_resampling,
                                    max_inchikey_sampling=max_inchikey_sampling
                                    )
    selected_inchikey_pairs = select_compound_pairs_wrapper(spectrums, settings)
    inchikey_pair_generator = InchikeyPairGenerator(selected_inchikey_pairs)

    highest_inchikey_count = max(inchikey_pair_generator.get_inchikey_counts().values())
    assert highest_inchikey_count <= max_inchikey_sampling + 1 # +1 because there is a chance that the last added inchikey is a pair to itself...


def check_correct_oversampling(selected_inchikey_pairs: InchikeyPairGenerator, max_resampling: int):
    pair_counts = Counter(selected_inchikey_pairs.selected_inchikey_pairs)
    for count in pair_counts.values():
        assert count <= max_resampling, "the resampling was done too frequently"


def get_available_score_distribution(settings, spectra):
    """Gets the score distribution for the available pairs (before doing balanced selection)"""
    fingerprints, inchikeys14_unique = compute_fingerprints_for_training(spectra, settings.fingerprint_type,
                                                                         settings.fingerprint_nbits)

    available_pairs_per_bin_matrix, available_scores_per_bin_matrix = compute_jaccard_similarity_per_bin(
        fingerprints,
        settings.max_pairs_per_bin,
        settings.same_prob_bins,
        settings.include_diagonal)

    score_distribution_per_inchikey = {inchikey: [0]*len(settings.same_prob_bins) for inchikey in inchikeys14_unique}
    for bin_id, available_pairs in enumerate(available_pairs_per_bin_matrix):
        for inchikey_1_idx, row_of_pairs in enumerate(available_pairs):
            inchikey_1 = inchikeys14_unique[inchikey_1_idx]
            score_distribution_per_inchikey[inchikey_1][bin_id] = len(np.where(row_of_pairs != -1)[0])
    return score_distribution_per_inchikey


def print_balanced_bins_per_inchikey(selected_inchikey_pairs: InchikeyPairGenerator, settings, spectra):
    """Prints the available distribution and the balanced distribution

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
            _ = balanced_distribution.index(min(balanced_distribution))
            print(available_distribution, balanced_distribution)
            # assert minimum_available_distribution*settings.max_pair_resampling == min(balanced_distribution)


def check_balanced_scores_selecting_inchikey_pairs(selected_inchikey_pairs: InchikeyPairGenerator,
                                                   score_bins):
    """Test if InchikeyPairGenerator has an equal inchikey distribution
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

