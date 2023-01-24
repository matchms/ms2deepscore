import numpy as np
import pytest

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from tests.test_user_worfklow import load_processed_spectrums, get_reference_scores


def create_test_data():
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)
    return binned_spectrums, tanimoto_scores_df


def test_DataGeneratorAllInchikeys():
    """Basic first test for DataGeneratorAllInchikeys"""
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
    assert A[0].shape == A[1].shape == (10, 88), "Expected different data shape"
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
                                0.50632911, 0.5214447, 0.52663934, 0.59934853, 0.63581489])
    repetitions = 200
    collect_results = np.zeros(repetitions * batch_size)  # Collect 2000 results
    for i in range(repetitions):
        _, B = test_generator.__getitem__(0)
        collect_results[batch_size*i:batch_size*(i+1)] = B
    assert len(np.unique(collect_results)) <= 15, "Expected max 15 possible results"
    present_in_expected_labels = [(np.round(x, 6) in list(np.round(expected_labels, 6)))
                                  for x in np.unique(collect_results)]
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

    # Create normal generator
    normal_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                 reference_scores_df=tanimoto_scores_df,
                                                 dim=dimension, batch_size=batch_size,
                                                 use_fixed_set=False)

    # Create generator that generates a fixed set every epoch
    fixed_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                reference_scores_df=tanimoto_scores_df,
                                                dim=dimension, batch_size=batch_size,
                                                num_turns=5, use_fixed_set=True)

    def collect_results(generator):
        n_batches = len(generator)
        X = np.zeros((batch_size, dimension, 2, n_batches))
        y = np.zeros((batch_size, n_batches))
        for i, batch in enumerate(generator):
            X[:, :, 0, i] = batch[0][0]
            X[:, :, 1, i] = batch[0][1]
            y[:, i] = batch[1]
        return X, y

    first_X, first_y = collect_results(normal_generator)
    second_X, second_y = collect_results(normal_generator)
    assert not np.array_equal(first_X, second_X)
    assert first_y.shape == (4, 2), "Expected different number of labels"

    first_X, first_y = collect_results(fixed_generator)
    second_X, second_y = collect_results(fixed_generator)
    assert np.array_equal(first_X, second_X)
    assert first_y.shape == (4, 10), "Expected different number of labels"

    # Create another fixed generator based on the same dataset that should generate the same
    # fixed set
    fixed_generator2 = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
                                                 reference_scores_df=tanimoto_scores_df,
                                                 dim=dimension, batch_size=batch_size,
                                                 num_turns=5, use_fixed_set=True)
    first_X, first_y = collect_results(fixed_generator)
    second_X, second_y = collect_results(fixed_generator2)
    assert np.array_equal(first_X, second_X)


def test_DataGeneratorAllSpectrums_additional_inputs():
    """
    Test if additional input parameter works as intended 
    """

    # Get test data
    additional_input = {"precursor_mz": 0.001, "parent_mass": 0.001}
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                  additional_metadata=additional_input)
    binned_spectrums = ms2ds_binner.fit_transform(spectrums)

    # Define other parameters
    batch_size = 4
    dimension = 88
    data_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                               dim=dimension, additional_input=additional_input)
    batch_X, batch_y = data_generator.__getitem__(0)

    assert len(batch_X) != len(batch_y), "Batchsizes from X and y are not the same."
    assert len(batch_X[0]) != 3, "There are not as many inputs as specified."
