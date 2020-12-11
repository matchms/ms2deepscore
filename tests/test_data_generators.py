import os
import json
import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
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