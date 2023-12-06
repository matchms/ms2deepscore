import string
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import (DataGeneratorAllInchikeys,
                                          DataGeneratorAllSpectrums,
                                          DataGeneratorCherrypicked,
                                          _exclude_nans_from_labels,
                                          _validate_labels)
from ms2deepscore.MetadataFeatureGenerator import (CategoricalToBinary,
                                                   StandardScaler)
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper
from tests.test_user_worfklow import (get_reference_scores,
                                      load_processed_spectrums)


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


def create_test_spectra(num_of_unique_inchikeys):
    # Define other parameters
    mz, intens = 100.0, 0.1
    spectrums = []
    letters = list(string.ascii_uppercase[:num_of_unique_inchikeys])
    letters += letters

    def generate_binary_vector(i):
        binary_vector = np.zeros(10, dtype=int)
        binary_vector[i % 3] = 1
        binary_vector[i % 5 + 3] = 1
        binary_vector[i % 4] = 1
        binary_vector[i % 10] = 1
        binary_vector[8 - i // 9] = 1
        binary_vector[6 - i // 15] = 1
        return binary_vector

    # Create fake spectra
    fake_inchikeys = []
    for i, letter in enumerate(letters):
        dummy_inchikey = f"{14 * letter}-{10 * letter}-N"
        fingerprint = generate_binary_vector(i)
        fake_inchikeys.append(dummy_inchikey)
        spectrums.append(Spectrum(mz=np.array([mz + (i+1) * 25.0]), intensities=np.array([intens]),
                                metadata={"precursor_mz": 111.1,
                                            "inchikey": dummy_inchikey,
                                            "compound_name": letter,
                                            "fingerprint": fingerprint,
                                            }))
    return spectrums


def test_DataGeneratorCherrypicked():
    """Test DataGeneratorCherrypicked using generated data.
    """
    num_of_unique_inchikeys = 15
    spectrums = create_test_spectra(num_of_unique_inchikeys)
    batch_size = 8

    ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=1)
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)
    dimension = len(ms2ds_binner.known_bins)
    settings = SettingsMS2Deepscore({"tanimoto_bins": np.array([(x / 4, x / 4 + 0.25) for x in range(0, 4)]),
                                     "average_pairs_per_bin": 1})
    scp, spectrums = select_compound_pairs_wrapper(spectrums, settings)
    # Create generator
    test_generator = DataGeneratorCherrypicked(binned_spectrums=binned_spectrums,
                                               spectrum_binner=ms2ds_binner,
                                               selected_compound_pairs=scp,
                                               batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0,
                                               augment_noise_max=0)

    x, y = test_generator.__getitem__(0)
    assert x[0].shape == x[1].shape == (batch_size, dimension), "Expected different data shape"
    assert y.shape[0] == batch_size
    assert set(test_generator.indexes) == set(list(range(num_of_unique_inchikeys))), "Something wrong with generator indices"

    # Test many cycles --> scores properly distributed into bins?
    counts = []
    repetitions = 100
    total = batch_size * repetitions
    for _ in range(repetitions):
        for i, batch in enumerate(test_generator):
            counts.extend(list(batch[1]))
    assert len(counts) == total
    assert (np.array(counts) > 0.5).sum() > 0.4 * total
    assert (np.array(counts) <= 0.5).sum() > 0.4 * total
    # Check mostly equal distribution accross all four bins:
    assert (np.array(counts) <= 0.25).sum() > 0.22 * total
    assert ((np.array(counts) > 0.25) & (np.array(counts) <= 0.5)).sum() > 0.22 * total
    assert ((np.array(counts) > 0.5) & (np.array(counts) <= 0.75)).sum() > 0.22 * total
    assert (np.array(counts) > 0.75).sum() > 0.22 * total


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


def test_DataGeneratorAllSpectrums():
    """Basic first test for DataGeneratorAllInchikeys using actual data.
    """
    binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_dummy_data()
    assert binned_spectrums[0].binned_peaks == {0: 0.1}, "Something went wrong with the binning"

    # Define other parameters
    batch_size = 8 # Set the batch size to 8 to make sure it is a different number than the number of bins.
    dimension = len(ms2ds_binner.known_bins)
    # Create generator
    test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums,
                                               reference_scores_df=tanimoto_scores_df,
                                               spectrum_binner=ms2ds_binner,
                                               batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0,
                                               augment_noise_max=0,
                                               random_seed=41)

    x, y = test_generator.__getitem__(0)
    assert x[0].shape == x[1].shape == (batch_size, dimension), "Expected different data shape"
    assert set(test_generator.indexes) == set(list(range(len(binned_spectrums)))), "Something wrong with generator indices"

    # Test if every inchikey not was picked only once
    assert not (x[0].sum(axis=0) > 0).sum() == batch_size, \
        "For each inchikey only one spectrum was picked instead of all spectra"

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
    assert test_generator.settings.num_turns == 1, "Expected different default."
    assert test_generator.settings.augment_intensity == 0.0, "Expected changed value."


def test_DataGeneratorAllSpectrumsRealData():
    """Basic first test for DataGeneratorAllSpectrums"""
    # Get test data
    binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()

    # Define other parameters
    batch_size = 10
    dimension = 88

    # Create generator
    test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:150],
                                               reference_scores_df=tanimoto_scores_df,
                                               spectrum_binner=ms2ds_binner, batch_size=batch_size,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0)

    A, B = test_generator.__getitem__(0)
    assert A[0].shape == A[1].shape == (10, dimension), "Expected different data shape"
    assert B.shape[0] == 10, "Expected different label shape."
    assert test_generator.settings.num_turns == 1, "Expected different default."
    assert test_generator.settings.augment_intensity == 0.0, "Expected changed value."


def test_DataGeneratorAllSpectrums_no_inchikey_leaking():
    """Test if non-selected InChIKeys are correctly removed"""
    # Get test data
    binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()

    # Define other parameters
    batch_size = 6
    dimension = 88

    # Create generator
    test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                               reference_scores_df=tanimoto_scores_df,
                                               spectrum_binner=ms2ds_binner,
                                               batch_size=batch_size,
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
    binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()
    asymmetric_scores_df = tanimoto_scores_df.iloc[:, 2:]
    with pytest.raises(ValueError) as msg:
        _ = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums,
                                      reference_scores_df=asymmetric_scores_df,
                                      spectrum_binner=ms2ds_binner)
    assert "index and columns of reference_scores_df are not identical" in str(msg), \
        "Expected different ValueError"


def test_DataGeneratorAllSpectrums_fixed_set():
    """
    Test whether use_fixed_set=True toggles generating the same dataset on each epoch.
    """
    # Get test data
    binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()

    # Define other parameters
    batch_size = 4
    dimension = 88

    # Create generator that generates a fixed set every epoch
    fixed_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                reference_scores_df=tanimoto_scores_df,
                                                spectrum_binner=ms2ds_binner, batch_size=batch_size,
                                                num_turns=5, use_fixed_set=True)

    first_X, first_y = collect_results(fixed_generator, batch_size, dimension)
    second_X, second_y = collect_results(fixed_generator, batch_size, dimension)
    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_y, second_y)
    assert fixed_generator.settings.random_seed is None


def test_DataGeneratorAllSpectrums_fixed_set_random_seed():
    """
    Test whether use_fixed_set=True toggles generating the same dataset on each epoch.
    And if same random_seed leads to exactly the same output.
    """
    # Get test data
    binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()

    # Define other parameters
    batch_size = 4
    dimension = 88

    # Create normal generator
    normal_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                 reference_scores_df=tanimoto_scores_df,
                                                 spectrum_binner=ms2ds_binner, batch_size=batch_size,
                                                 use_fixed_set=False)

    # Create generator that generates a fixed set every epoch
    fixed_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                reference_scores_df=tanimoto_scores_df,
                                                spectrum_binner=ms2ds_binner, batch_size=batch_size,
                                                num_turns=5, use_fixed_set=True, random_seed=0)

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
                                                 spectrum_binner=ms2ds_binner, batch_size=batch_size,
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

    # Run for two test cases.
    # Testing a single and multiple inputs is important, since numpy can do weird things with 1D arrays of len= 1
    test_cases = [(StandardScaler("precursor_mz", mean=0, std=1000), ),
                  (StandardScaler("precursor_mz", mean=0, std=1000),
                   CategoricalToBinary("ionmode", entries_becoming_one="negative", entries_becoming_zero="positive"))]
    for additional_feature_types in test_cases:

        # additional_feature_types = ()
        ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                      additional_metadata=additional_feature_types)
        binned_spectrums = ms2ds_binner.fit_transform(spectrums)

        # Define other parameters
        batch_size = 4
        data_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                                   spectrum_binner=ms2ds_binner, batch_size=batch_size)
        batch_X, batch_y = data_generator.__getitem__(0)
        for batch_X_values in batch_X:
            assert len(batch_X_values) == len(batch_y) == batch_size, "Batchsizes from X and y are not the same."
        assert len(batch_X[1][0]) == len(additional_feature_types) == len(batch_X[3][0]), "There are not as many inputs as specified."



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
