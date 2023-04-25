import numpy as np
import pandas as pd
import pytest
import string
from matchms import Spectrum

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import (DataGeneratorAllInchikeys,
                                          DataGeneratorAllSpectrums,
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
        spectrums.append(Spectrum(mz=np.array([mz + (i+1) * 25.0]), intensities=np.array([2*intens]),
                                  metadata={"inchikey": dummy_inchikey,
                                            "compound_name": f"{letter}-2"}))

    # Set the column and index names
    tanimoto_fake.columns = [x[:14] for x in fake_inchikeys]
    tanimoto_fake.index = [x[:14] for x in fake_inchikeys]

    ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=1)
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)
    return binned_spectrums, tanimoto_fake


def create_test_data():
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)
    return binned_spectrums, tanimoto_scores_df


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
    binned_spectrums, tanimoto_scores_df = create_dummy_data()

    # Define other parameters
    batch_size = 10
    dimension = tanimoto_scores_df.shape[0]

    selected_inchikeys = tanimoto_scores_df.index
    # Create generator
    test_generator = DataGeneratorAllInchikeys(binned_spectrums=binned_spectrums,
                                                selected_inchikeys=selected_inchikeys,
                                                reference_scores_df=tanimoto_scores_df,
                                                dim=dimension, batch_size=batch_size,
                                                augment_removal_max=0.0,
                                                augment_removal_intensity=0.0,
                                                augment_intensity=0.0,
                                                augment_noise_max=0)

    A, B = test_generator.__getitem__(0)
    assert binned_spectrums[0].binned_peaks == {0: 0.1}, "Something went wrong with the binning"
    assert A[0].shape == A[1].shape == (batch_size, dimension), "Expected different data shape"
    assert set(test_generator.indexes) == set(list(range(10))), "Something wrong with generator indices"

    # Test if every inchikey was picked once (and only once):
    assert (A[0] > 0).sum() == 10
    assert np.all((A[0] > 0).sum(axis=1) == (A[0] > 0).sum(axis=0))

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
    binned_spectrums, tanimoto_scores_df = create_test_data()

    # Define other parameters
    batch_size = 10
    dimension = 88

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
    assert A[0].shape == A[1].shape == (batch_size, dimension), "Expected different data shape"
    assert B.shape[0] == 10, "Expected different label shape."
    assert test_generator.settings["num_turns"] == 1, "Expected different default."
    assert test_generator.settings["augment_intensity"] == 0.0, "Expected changed value."


def test_DataGeneratorAllSpectrums():
    """Basic first test for DataGeneratorAllSpectrums"""
    # Get test data
    binned_spectrums, tanimoto_scores_df = create_test_data()

    # Define other parameters
    batch_size = 10
    dimension = 88

    # Create generator
    test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:150],
                                               reference_scores_df=tanimoto_scores_df,
                                               dim=dimension, batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0)

    A, B = test_generator.__getitem__(0)
    assert A[0].shape == A[1].shape == (10, 88), "Expected different data shape"
    assert B.shape[0] == 10, "Expected different label shape."
    assert test_generator.settings["num_turns"] == 1, "Expected different default."
    assert test_generator.settings["augment_intensity"] == 0.0, "Expected changed value."


def test_DataGeneratorAllSpectrums_no_inchikey_leaking():
    """Test if non-selected InChIKeys are correctly removed"""
    # Get test data
    binned_spectrums, tanimoto_scores_df = create_test_data()

    # Define other parameters
    batch_size = 8
    dimension = 88

    # Create generator
    test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                               reference_scores_df=tanimoto_scores_df,
                                               dim=dimension, batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0)

    assert test_generator.reference_scores_df.shape == (6, 6), "Expected different reduced shape of labels"
    expected_inchikeys = ['BBXXLROWFHWFQY',
                          'FBOUIAKEJMZPQG',
                          'GPXLRLUVLMHHIK',
                          'JXCGFZXSOMJFOA',
                          'RZILCCPWPBTYDO',
                          'UYJUZNLFJAWNEZ']
    found_inchikeys = test_generator.reference_scores_df.columns.to_list()
    found_inchikeys.sort()
    assert found_inchikeys == expected_inchikeys, \
        "Expected different InChIKeys to remain in reference_scores_df"

    # Test if the expected labels are returned by generator
    expected_labels = np.array([0.38944724, 0.39130435, 0.39378238, 0.40045767, 0.40497738,
                                0.40930233, 0.43432203, 0.46610169, 0.47416413, 0.48156182,
                                0.50632911, 0.5214447 , 0.52663934, 0.59934853, 0.63581489])
    repetitions = 200
    collect_results = np.zeros(repetitions * batch_size)  # Collect 2000 results
    for i in range(repetitions):
        _, B = test_generator.__getitem__(0)
        collect_results[batch_size*i:batch_size*(i+1)] = B
    assert len(np.unique(collect_results)) <= 15, "Expected max 15 possible results"
    present_in_expected_labels = [(np.round(x,6) in list(np.round(expected_labels, 6))) for x in np.unique(collect_results)]
    assert np.all(present_in_expected_labels), "Got unexpected labels from generator"


def test_DataGeneratorAllSpectrums_asymmetric_label_input():
    # Create generator
    binned_spectrums, tanimoto_scores_df = create_test_data()
    asymmetric_scores_df = tanimoto_scores_df.iloc[:, 2:]
    with pytest.raises(ValueError) as msg:
        _ = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums,
                                      reference_scores_df=asymmetric_scores_df,
                                      dim=101)
    assert "index and columns of reference_scores_df are not identical" in str(msg), \
        "Expected different ValueError"


def test_DataGeneratorAllSpectrums_fixed_set():
    """
    Test whether use_fixed_set=True toggles generating the same dataset on each epoch.
    """
    # Get test data
    binned_spectrums, tanimoto_scores_df = create_test_data()

    # Define other parameters
    batch_size = 4
    dimension = 88

    # Create generator that generates a fixed set every epoch
    fixed_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                reference_scores_df=tanimoto_scores_df,
                                                dim=dimension, batch_size=batch_size,
                                                num_turns=5, use_fixed_set=True)

    first_X, first_y = collect_results(fixed_generator, batch_size, dimension)
    second_X, second_y = collect_results(fixed_generator, batch_size, dimension)
    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_y, second_y)
    assert fixed_generator.settings["random_seed"] is None


def test_DataGeneratorAllSpectrums_fixed_set_random_seed():
    """
    Test whether use_fixed_set=True toggles generating the same dataset on each epoch.
    And if same random_seed leads to exactly the same output.
    """
    # Get test data
    binned_spectrums, tanimoto_scores_df = create_test_data()

    # Define other parameters
    batch_size = 4
    dimension = 88

    # Create normal generator
    normal_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                 reference_scores_df=tanimoto_scores_df,
                                                 dim=dimension, batch_size=batch_size,
                                                 use_fixed_set=False)

    # Create generator that generates a fixed set every epoch
    fixed_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                reference_scores_df=tanimoto_scores_df,
                                                dim=dimension, batch_size=batch_size,
                                                num_turns=5, use_fixed_set=True,
                                                random_seed=0)

    first_X, first_y = collect_results(normal_generator, batch_size, dimension)
    second_X, second_y = collect_results(normal_generator, batch_size, dimension)
    assert not np.array_equal(first_X, second_X)
    assert first_y.shape == (4, 2), "Expected different number of labels"

    first_X, first_y = collect_results(fixed_generator, batch_size, dimension)
    second_X, second_y = collect_results(fixed_generator, batch_size, dimension)
    assert np.array_equal(first_X, second_X)
    assert first_y.shape == (4, 10), "Expected different number of labels"

    # Create another fixed generator based on the same dataset that should generate the same
    # fixed set
    fixed_generator2 = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                 reference_scores_df=tanimoto_scores_df,
                                                 dim=dimension, batch_size=batch_size,
                                                 num_turns=5, use_fixed_set=True,
                                                 random_seed=0)
    first_X, first_y = collect_results(fixed_generator, batch_size, dimension)
    second_X, second_y = collect_results(fixed_generator2, batch_size, dimension)
    assert np.array_equal(first_X, second_X)


def test_DataGeneratorAllSpectrums_additional_inputs():
    """
    Test if additional input parameter works as intended 
    """
    
    # Get test data
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5, 
                                    additional_metadata=["parent_mass", "precursor_mz"])
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)


    # Define other parameters
    batch_size = 4
    dimension = 88
    additional_input=["precursor_mz", "parent_mass"]
    data_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                           dim=dimension, additional_input=additional_input)
    batch_X, batch_y = data_generator.__getitem__(0)

    assert len(batch_X) != len(batch_y), "Batchsizes from X and y are not the same."
    assert len(batch_X[0]) != 3, "There are not as many inputs as specified."


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
