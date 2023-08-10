import numpy as np
import pandas as pd
import pytest
import string
from matchms import Spectrum

from ms2deepscore import SpectrumBinner
from ms2deepscore.MetadataFeatureGenerator import StandardScaler, CategoricalToBinary
from ms2deepscore.data_generators import (DataGeneratorAllInchikeys,
                                          _exclude_nans_from_labels,
                                          _validate_labels)
from tests.test_user_worfklow import load_processed_spectrums, get_reference_scores


def create_dummy_data():
    """Create fake data to test generators.
    """
    mz, intens = 100.0, 0.1
    spectrums = []

    letters = list(string.ascii_uppercase[:10])

    # Create fake similarities
    similarities = {}
    for i, letter1 in enumerate(letters):
        for j, letter2 in enumerate(letters):
            similarities[(letter1, letter2)] = (len(letters) - abs(i - j)) / len(letters)

    tanimoto_fake = pd.DataFrame(similarities.values(),
                                 index=similarities.keys()).unstack()

    # Create fake spectra
    fake_inchikeys = []
    for i, letter in enumerate(letters):
        dummy_inchikey = f"{14 * letter}-{10 * letter}-N"
        fake_inchikeys.append(dummy_inchikey)
        spectrums.append(Spectrum(mz=np.array([mz + (i+1) * 25.0]), intensities=np.array([intens]),
                                  metadata={"inchikey": dummy_inchikey,
                                            "compound_name": letter}))
        # Generate a duplicated spectrum for half the inchikeys
        if i >= 5:
            spectrums.append(Spectrum(mz=np.array([mz + (i+1) * 25.0]), intensities=np.array([2*intens]),
                                      metadata={"inchikey": dummy_inchikey,
                                                "compound_name": f"{letter}-2"}))

    # Set the column and index names
    tanimoto_fake.columns = [x[:14] for x in fake_inchikeys]
    tanimoto_fake.index = [x[:14] for x in fake_inchikeys]

    ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=1)
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)
    return binned_spectrums, tanimoto_fake, ms2ds_binner


def create_test_data():
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)
    return binned_spectrums, tanimoto_scores_df, ms2ds_binner


def collect_results(generator, batch_size, dimension):
    n_batches = len(generator)
    X = np.zeros((batch_size, dimension, 2, n_batches))
    y = np.zeros((batch_size, n_batches))
    for i, batch in enumerate(generator):
        X[:, :, 0, i] = batch[0][0]
        X[:, :, 1, i] = batch[0][1]
        y[:, i] = batch[1]
    return X, y


def test_DataGeneratorAllInchikeys():
    """Test DataGeneratorAllInchikeys using generated data.
    """
    binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_dummy_data()
    assert binned_spectrums[0].binned_peaks == {0: 0.1}, "Something went wrong with the binning"

    # Define other parameters
    batch_size = 8
    dimension = len(ms2ds_binner.known_bins)

    selected_inchikeys = tanimoto_scores_df.index
    # Create generator
    test_generator = DataGeneratorAllInchikeys(binned_spectrums=binned_spectrums,
                                               selected_inchikeys=selected_inchikeys,
                                               spectrum_binner=ms2ds_binner,
                                               reference_scores_df=tanimoto_scores_df,
                                               batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0,
                                               augment_noise_max=0)

    x, y = test_generator.__getitem__(0)
    assert x[0].shape == x[1].shape == (batch_size, dimension), "Expected different data shape"
    assert set(test_generator.indexes) == set(list(range(dimension))), "Something wrong with generator indices"

    # Test if every inchikey was picked once (and only once):
    assert (x[0].sum(axis=0) > 0).sum() == batch_size # This works since each spectrum in the dummy set has one unique peak

    # Test many cycles --> scores properly distributed into bins?
    counts = []
    repetitions = 100
    total = batch_size * repetitions
    for _ in range(repetitions):
        for i, batch in enumerate(test_generator):
            counts.extend(list(batch[1]))
    assert (np.array(counts) > 0.5).sum() > 0.4 * total
    assert (np.array(counts) <= 0.5).sum() > 0.4 * total


def test_DataGeneratorAllInchikeys_real_data():
    """Basic first test for DataGeneratorAllInchikeys using actual data.
    """
    # Get test data
    binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()

    # Define other parameters
    batch_size = 10
    dimension = len(ms2ds_binner.known_bins)

    selected_inchikeys = tanimoto_scores_df.index[:80]
    # Create generator
    test_generator = DataGeneratorAllInchikeys(binned_spectrums=binned_spectrums,
                                               selected_inchikeys=selected_inchikeys,
                                               reference_scores_df=tanimoto_scores_df,
                                               spectrum_binner=ms2ds_binner, batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0)

    A, B = test_generator.__getitem__(0)
    assert A[0].shape == A[1].shape == (batch_size, dimension), "Expected different data shape"
    assert B.shape[0] == 10, "Expected different label shape."
    assert test_generator.settings["num_turns"] == 1, "Expected different default."
    assert test_generator.settings["augment_intensity"] == 0.0, "Expected changed value."


# Test specific class methods
# ---------------------------
def test_validate_labels():
    # Test case 1: reference_scores_df with different index and column names
    ref_scores = pd.DataFrame({'A1': [0.5, 0.6], 'A2': [0.7, 0.8]}, index=['B1', 'B2'])
    with pytest.raises(ValueError):
        _validate_labels(ref_scores)

    # Test case 2: reference_scores_df with identical index and column names
    ref_scores = pd.DataFrame({'A1': [0.5, 0.6], 'A2': [0.7, 0.8]}, index=['A1', 'A2'])
    _validate_labels(ref_scores)  # Should not raise ValueError


def test_exclude_nans_from_labels():
    # Create a sample DataFrame with NaN values
    data = {
        "A": [1, 2, np.nan, 4],
        "B": [2, 3, 4, 5],
        "C": [3, 4, 5, np.nan],
        "D": [4, 5, 6, 7]
    }
    reference_scores_df = pd.DataFrame(data, index=["A", "B", "C", "D"])

    # Call the _exclude_nans_from_labels method
    clean_df = _exclude_nans_from_labels(reference_scores_df)

    # Expected DataFrame after removing rows and columns with NaN values
    expected_data = {
        "A": [1, 2],
        "B": [2, 3]
    }
    expected_clean_df = pd.DataFrame(expected_data, index=["A", "B"])

    # Check if the cleaned DataFrame is equal to the expected DataFrame
    assert np.allclose(clean_df.values, expected_clean_df.values)
    assert np.all(clean_df.index == clean_df.columns)
    assert np.all(clean_df.index == ["A", "B"])
