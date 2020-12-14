import os
import json
import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
path_tests  = os.path.dirname(__file__)


def create_test_data():
    spectrums_binned_file = os.path.join(path_tests, "testdata_spectrums_binned.json")
    with open(spectrums_binned_file, "r") as read_file:
        spectrums_binned = json.load(read_file)

    score_array = np.load(os.path.join(path_tests, "testdata_tanimoto_scores.npy"))
    inchikeys_array = np.load(os.path.join(path_tests, "testdata_inchikeys.npy"))
    inchikey_score_mapping = np.load(os.path.join(path_tests, "testdata_inchikey_score_mapping.npy"),
                                     allow_pickle=True)
    return spectrums_binned, score_array, inchikey_score_mapping, inchikeys_array


def test_DataGeneratorAllInchikeys():
    """Basic first test for DataGeneratorAllInchikeys"""
    # Get test data
    spectrums_binned, score_array, inchikey_score_mapping, inchikeys_all = create_test_data()

    # Define other parameters
    batch_size = 10
    num_turns = 1
    peak_scaling = 0.5
    dimension = 101
    same_prob_bins = [(0, 0.5), (0.5, 1)]

    inchikey_ids = np.arange(0,80)

    # Create generator
    test_generator = DataGeneratorAllInchikeys(spectrums_binned, score_array, inchikey_ids,
                                               inchikey_score_mapping, inchikeys_all,
                                               dim=dimension, batch_size=batch_size,
                                               num_turns=num_turns, peak_scaling=peak_scaling,
                                               shuffle=True, ignore_equal_pairs=True,
                                               same_prob_bins=same_prob_bins,
                                               augment_peak_removal_max=0.0,
                                               augment_peak_removal_intensity=0.0,
                                               augment_intensity=0.0)

    A, B = test_generator.__getitem__(0)
    assert A[0].shape == A[1].shape == (10, 101), "Expected different data shape"
    assert B.shape[0] == 10, "Expected different label shape."


def test_DataGeneratorAllSpectrums():
    """Basic first test for DataGeneratorAllSpectrums"""
    # Get test data
    spectrums_binned, score_array, inchikey_score_mapping, inchikeys_all = create_test_data()

    # Define other parameters
    batch_size = 10
    num_turns = 1
    peak_scaling = 0.5
    dimension = 101
    same_prob_bins = [(0, 0.5), (0.5, 1)]

    spectrum_ids = np.arange(0,150)

    # Create generator
    test_generator = DataGeneratorAllSpectrums(spectrums_binned, score_array, spectrum_ids,
                                               inchikey_score_mapping, inchikeys_all,
                                               dim=dimension, batch_size=batch_size,
                                               num_turns=num_turns, peak_scaling=peak_scaling,
                                               shuffle=True, ignore_equal_pairs=True,
                                               same_prob_bins=same_prob_bins,
                                               augment_peak_removal_max=0.0,
                                               augment_peak_removal_intensity=0.0,
                                               augment_intensity=0.0)

    A, B = test_generator.__getitem__(0)
    assert A[0].shape == A[1].shape == (10, 101), "Expected different data shape"
    assert B.shape[0] == 10, "Expected different label shape."

def test_DataGeneratorAllSpectrums_input_error():
    """Test if expected error is raised for incorrect input formats"""
    # Get test data
    spectrums_binned, score_array, inchikey_score_mapping, inchikeys_all = create_test_data()

    # Define other parameters
    batch_size = 10
    num_turns = 1
    peak_scaling = 0.5
    dimension = 101
    same_prob_bins = [(0, 0.5), (0.5, 1)]

    spectrum_ids = np.arange(0,150)

    # Create generator --> wrong score array size
    with pytest.raises(AssertionError) as msg:
        _ = DataGeneratorAllSpectrums(spectrums_binned, score_array[:-2, :-2], spectrum_ids,
                                    inchikey_score_mapping, inchikeys_all,
                                    dim=dimension, batch_size=batch_size,
                                    num_turns=num_turns, peak_scaling=peak_scaling,
                                    shuffle=True, ignore_equal_pairs=True,
                                    same_prob_bins=same_prob_bins,
                                    augment_peak_removal_max=0.0,
                                    augment_peak_removal_intensity=0.0,
                                    augment_intensity=0.0)
    assert 'Expected score_array of size 100x100.' in str(msg.value), \
        "Expected different expection to be raised"

    # Create generator --> wrong score array size
    with pytest.raises(AssertionError) as msg:
        _ = DataGeneratorAllSpectrums(spectrums_binned[:-2], score_array, spectrum_ids,
                                    inchikey_score_mapping, inchikeys_all,
                                    dim=dimension, batch_size=batch_size,
                                    num_turns=num_turns, peak_scaling=peak_scaling,
                                    shuffle=True, ignore_equal_pairs=True,
                                    same_prob_bins=same_prob_bins,
                                    augment_peak_removal_max=0.0,
                                    augment_peak_removal_intensity=0.0,
                                    augment_intensity=0.0)
    assert "inchikeys_all and spectrums_binned must have the same dimension." in str(msg.value), \
        "Expected different exception to be raised"
