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

    # Create generator
    test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:150],
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


def test_DataGeneratorAllSpectrums_no_inchikey_leaking():
    """Test if non-selected InChIKeys are correctly removed"""
    # Get test data
    binned_spectrums, tanimoto_scores_df = create_test_data()

    # Define other parameters
    batch_size = 10
    dimension = 101

    # Create generator
    test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:11],
                                               labels_df=tanimoto_scores_df,
                                               dim=dimension, batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0)

    assert test_generator.labels_df.shape == (6, 6), "Expected different reduced shape of labels"
    expected_inchikeys = ['AAWZDTNXLSGCEK-TUNDHVGDSA-N',
                          'CXVGEDCSTKKODG-UHFFFAOYSA-N',
                          'JFFHVIUZNPTGGR-WJLGXSQGSA-N',
                          'JGCSKOVQDXEQHI-UHFFFAOYSA-N',
                          'VCBNPTWPJQLHQN-NYAJDEOCSA-N',
                          'ZBAMSLOMNLECFR-IEAZIUSSSA-N']
    found_inchikeys = test_generator.labels_df.columns.to_list()
    found_inchikeys.sort()
    assert found_inchikeys == expected_inchikeys, \
        "Expected different InChIKeys to remain in labels_df"

    # Test if the expected labels are returned by generator
    expected_labels = np.array([0.09285714, 0.11022727, 0.15672306, 0.15920916, 0.19264588,
                                0.20079523, 0.20326679, 0.21044304, 0.24236453, 0.25663717,
                                0.27233429, 0.27994122, 0.29661684, 0.41184669, 0.53772684])
    collect_results = np.zeros(2000)  # Collect 2000 results
    for i in range(200):
        _, B = test_generator.__getitem__(0)
        collect_results[batch_size*i:batch_size*(i+1)] = B
    assert len(np.unique(collect_results)) <= 15, "Expected max 15 possible results"
    present_in_expected_labels = [(np.round(x,6) in list(np.round(expected_labels, 6))) for x in np.unique(collect_results)]
    assert np.all(present_in_expected_labels), "Got unexpected labels from generator"


def test_DataGeneratorAllSpectrums_asymmetric_label_input():
    # Create generator
    binned_spectrums, tanimoto_scores_df = create_test_data()
    spectrum_ids = list(range(150))
    asymmetric_scores_df = tanimoto_scores_df.iloc[:, 2:]
    with pytest.raises(ValueError):
        test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums,
                                                   reference_scores_df=asymmetric_scores_df,
                                                   dim=101)
