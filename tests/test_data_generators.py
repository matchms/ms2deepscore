import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ms2deepscore import BinnedSpectrum
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.data_generators import DataGeneratorAllSpectrums

TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'

def create_test_data():
    binned_spectrums_file = TEST_RESOURCES_PATH / "testdata_spectrums_binned.json"
    with open(binned_spectrums_file, "r") as read_file:
        peaks_dicts = json.load(read_file)
    inchikeys_array = np.load(TEST_RESOURCES_PATH / "testdata_inchikeys.npy")
    binned_spectrums = []
    for i, peaks_dict in enumerate(peaks_dicts):
        binned_spectrums.append(BinnedSpectrum(binned_peaks=peaks_dict,
                                               metadata={"inchikey": inchikeys_array[i]}))

    tanimoto_scores_df = pd.read_csv(TEST_RESOURCES_PATH / 'testdata_tanimoto_scores.csv',
                                     index_col=0)
    return binned_spectrums, tanimoto_scores_df


def test_DataGeneratorAllInchikeys():
    """Basic first test for DataGeneratorAllInchikeys"""
    # Get test data
    binned_spectrums, tanimoto_scores_df = create_test_data()

    # Define other parameters
    batch_size = 10
    dimension = 101

    selected_inchikeys = tanimoto_scores_df.index[:80]
    # Create generator
    test_generator = DataGeneratorAllInchikeys(binned_spectrums=binned_spectrums,
                                               selected_inchikeys=selected_inchikeys,
                                               reference_scores_df=tanimoto_scores_df,
                                               dim=dimension, batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0)

    A, B = test_generator.__getitem__(0)
    assert A[0].shape == A[1].shape == (10, 101), "Expected different data shape"
    assert B.shape[0] == 10, "Expected different label shape."
    assert test_generator.settings["num_turns"] == 1, "Expected different default."
    assert test_generator.settings["augment_intensity"] == 0.0, "Expected changed value."


def test_DataGeneratorAllSpectrums():
    """Basic first test for DataGeneratorAllSpectrums"""
    # Get test data
    binned_spectrums, tanimoto_scores_df = create_test_data()

    # Define other parameters
    batch_size = 10
    dimension = 101

    spectrum_ids = list(range(150))

    # Create generator
    test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums,
                                               spectrum_ids=spectrum_ids,
                                               reference_scores_df=tanimoto_scores_df,
                                               dim=dimension, batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0)

    A, B = test_generator.__getitem__(0)
    assert A[0].shape == A[1].shape == (10, 101), "Expected different data shape"
    assert B.shape[0] == 10, "Expected different label shape."
    assert test_generator.settings["num_turns"] == 1, "Expected different default."
    assert test_generator.settings["augment_intensity"] == 0.0, "Expected changed value."


def test_DataGeneratorAllSpectrums_asymmetric_label_input():
    # Create generator
    binned_spectrums, tanimoto_scores_df = create_test_data()
    spectrum_ids = list(range(150))
    asymmetric_scores_df = tanimoto_scores_df.iloc[:, 2:]
    with pytest.raises(ValueError):
        test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums,
                                                   spectrum_ids=spectrum_ids,
                                                   reference_scores_df=asymmetric_scores_df,
                                                   dim=101)
