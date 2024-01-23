import os
import string
import numpy as np
import pandas as pd
import torch
from matchms import Spectrum
from ms2deepscore.data_generators import (DataGeneratorPytorch,
                                          TensorizationSettings,
                                          compute_validation_set,
                                          tensorize_spectra,
                                          write_to_pickle,
                                          load_generator_from_pickle)
from ms2deepscore.MetadataFeatureGenerator import (CategoricalToBinary,
                                                   StandardScaler)
from ms2deepscore.SettingsMS2Deepscore import \
    GeneratorSettings
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper
from tests.test_user_worfklow import (get_reference_scores,
                                      load_processed_spectrums)


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


def test_tensorize_spectra():
    spectrum = Spectrum(mz=np.array([10, 500, 999.9]), intensities=np.array([0.5, 0.5, 1]))
    tensorization_settings = TensorizationSettings(min_mz=10,
                                                   max_mz=1000,
                                                   mz_bin_width=1,
                                                   intensity_scaling=0.5,
                                                   additional_metadata=())
    spec_tensors, meta_tensors = tensorize_spectra([spectrum, spectrum], tensorization_settings)

    assert meta_tensors.shape == torch.Size([2, 0])
    assert spec_tensors.shape == torch.Size([2, 990])
    assert spec_tensors[0, 0] == spec_tensors[0, 490] == 0.5 ** 0.5
    assert spec_tensors[0, -1] == 1


def test_DataGeneratorPytorch():
    """Test DataGeneratorPytorch using generated data.
    """
    num_of_unique_inchikeys = 15
    spectrums = create_test_spectra(num_of_unique_inchikeys)
    batch_size = 8

    settings = GeneratorSettings({"same_prob_bins": np.array([(x / 4, x / 4 + 0.25) for x in range(0, 4)]),
                                  "average_pairs_per_bin": 1})
    scp, spectrums = select_compound_pairs_wrapper(spectrums, settings)
    tensorization_settings = TensorizationSettings(min_mz=10,
                                                   max_mz=1000,
                                                   mz_bin_width=0.1,
                                                   intensity_scaling=0.5,
                                                   additional_metadata=())
    # Create generator
    test_generator = DataGeneratorPytorch(
        spectrums=spectrums,
        tensorization_settings=tensorization_settings,
        selected_compound_pairs=scp,
        batch_size=batch_size,
        augment_removal_max=0.0,
        augment_removal_intensity=0.0,
        augment_intensity=0.0,
        augment_noise_max=0,
    )

    spec1, spec2, meta1, meta2, targets = test_generator.__getitem__(0)
    assert meta1.shape[0] == meta2.shape[0] == batch_size
    assert meta1.shape[1] == meta2.shape[1] == 0
    assert spec1.shape[0] == spec2.shape[0] == batch_size
    assert spec1.shape[1] == spec2.shape[1] == 9900
    assert targets.shape[0] == batch_size
    assert len(test_generator.indexes) == 15
    assert len(test_generator) == 2

    counts = []
    repetitions = 100
    total = num_of_unique_inchikeys * repetitions
    for _ in range(repetitions):
        for i, batch in enumerate(test_generator):
            counts.extend(batch[4])
    assert len(counts) == total
    assert (np.array(counts) > 0.5).sum() > 0.4 * total
    assert (np.array(counts) <= 0.5).sum() > 0.4 * total

    # Check mostly equal distribution across all four bins:
    assert (np.array(counts) <= 0.25).sum() > 0.22 * total
    assert ((np.array(counts) > 0.25) & (np.array(counts) <= 0.5)).sum() > 0.22 * total
    assert ((np.array(counts) > 0.5) & (np.array(counts) <= 0.75)).sum() > 0.22 * total
    assert (np.array(counts) > 0.75).sum() > 0.22 * total


def test_compute_validation_generator(tmp_path):


    num_of_unique_inchikeys = 15
    spectrums = create_test_spectra(num_of_unique_inchikeys)

    settings = GeneratorSettings({
        "same_prob_bins": np.array([(x / 2, x / 2 + 1/2) for x in range(0, 2)]),
        "average_pairs_per_bin": 2,
        "use_fixed_set": True,
        "batch_size": 5,
        "num_turns": 1
    })
    val_generator = compute_validation_set(spectrums, TensorizationSettings(), settings)
    generator_file = os.path.join(tmp_path, "generator.pickle")

    write_to_pickle(val_generator, generator_file)
    loaded_generator = load_generator_from_pickle(generator_file)
    batch_0 = val_generator.__getitem__(0)
    batch_0_saved = loaded_generator.__getitem__(0)
    assert len(batch_0) == 5 == len(batch_0_saved)
    for i, tensor in enumerate(batch_0):
        torch.equal(tensor, batch_0_saved[i])
    assert "spectrums" not in val_generator.__dict__, "Spectrums should have been removed"
    assert len(val_generator) == 3
    assert torch.allclose(batch_0[4], torch.tensor([0.5000, 0.2500, 0.4286, 0.4286, 0.1429]), atol=1e8)


#
#
# def test_DataGeneratorAllSpectrumsRealData():
#     """Basic first test for DataGeneratorAllSpectrums"""
#     # Get test data
#     binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()
#
#     # Define other parameters
#     batch_size = 10
#     dimension = 88
#
#     # Create generator
#     test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:150],
#                                                reference_scores_df=tanimoto_scores_df,
#                                                spectrum_binner=ms2ds_binner, batch_size=batch_size,
#                                                augment_removal_max=0.0,
#                                                augment_removal_intensity=0.0,
#                                                augment_intensity=0.0)
#
#     A, B = test_generator.__getitem__(0)
#     assert A[0].shape == A[1].shape == (10, dimension), "Expected different data shape"
#     assert B.shape[0] == 10, "Expected different label shape."
#     assert test_generator.settings.num_turns == 1, "Expected different default."
#     assert test_generator.settings.augment_intensity == 0.0, "Expected changed value."
#
#
# def test_DataGeneratorAllSpectrums_no_inchikey_leaking():
#     """Test if non-selected InChIKeys are correctly removed"""
#     # Get test data
#     binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()
#
#     # Define other parameters
#     batch_size = 6
#     dimension = 88
#
#     # Create generator
#     test_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
#                                                reference_scores_df=tanimoto_scores_df,
#                                                spectrum_binner=ms2ds_binner,
#                                                batch_size=batch_size,
#                                                augment_removal_max=0.0,
#                                                augment_removal_intensity=0.0,
#                                                augment_intensity=0.0)
#
#     assert test_generator.reference_scores_df.shape == (6, 6), "Expected different reduced shape of labels"
#     expected_inchikeys = ['BBXXLROWFHWFQY',
#                           'FBOUIAKEJMZPQG',
#                           'GPXLRLUVLMHHIK',
#                           'JXCGFZXSOMJFOA',
#                           'RZILCCPWPBTYDO',
#                           'UYJUZNLFJAWNEZ']
#     found_inchikeys = test_generator.reference_scores_df.columns.to_list()
#     found_inchikeys.sort()
#     assert found_inchikeys == expected_inchikeys, \
#         "Expected different InChIKeys to remain in reference_scores_df"
#
#     # Test if the expected labels are returned by generator
#     expected_labels = np.array([0.38944724, 0.39130435, 0.39378238, 0.40045767, 0.40497738,
#                                 0.40930233, 0.43432203, 0.46610169, 0.47416413, 0.48156182,
#                                 0.50632911, 0.5214447 , 0.52663934, 0.59934853, 0.63581489])
#     repetitions = 200
#     collect_results = np.zeros(repetitions * batch_size)  # Collect 2000 results
#     for i in range(repetitions):
#         _, B = test_generator.__getitem__(0)
#         collect_results[batch_size*i:batch_size*(i+1)] = B
#     assert len(np.unique(collect_results)) <= 15, "Expected max 15 possible results"
#     present_in_expected_labels = [(np.round(x,6) in list(np.round(expected_labels, 6))) for x in np.unique(collect_results)]
#     assert np.all(present_in_expected_labels), "Got unexpected labels from generator"
#
#
# def test_DataGeneratorAllSpectrums_asymmetric_label_input():
#     # Create generator
#     binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()
#     asymmetric_scores_df = tanimoto_scores_df.iloc[:, 2:]
#     with pytest.raises(ValueError) as msg:
#         _ = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums,
#                                       reference_scores_df=asymmetric_scores_df,
#                                       spectrum_binner=ms2ds_binner)
#     assert "index and columns of reference_scores_df are not identical" in str(msg), \
#         "Expected different ValueError"
#
#
# def test_DataGeneratorAllSpectrums_fixed_set():
#     """
#     Test whether use_fixed_set=True toggles generating the same dataset on each epoch.
#     """
#     # Get test data
#     binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()
#
#     # Define other parameters
#     batch_size = 4
#     dimension = 88
#
#     # Create generator that generates a fixed set every epoch
#     fixed_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
#                                                 reference_scores_df=tanimoto_scores_df,
#                                                 spectrum_binner=ms2ds_binner, batch_size=batch_size,
#                                                 num_turns=5, use_fixed_set=True)
#
#     first_X, first_y = collect_results(fixed_generator, batch_size, dimension)
#     second_X, second_y = collect_results(fixed_generator, batch_size, dimension)
#     assert np.array_equal(first_X, second_X)
#     assert np.array_equal(first_y, second_y)
#     assert fixed_generator.settings.random_seed is None
#
#
# def test_DataGeneratorAllSpectrums_fixed_set_random_seed():
#     """
#     Test whether use_fixed_set=True toggles generating the same dataset on each epoch.
#     And if same random_seed leads to exactly the same output.
#     """
#     # Get test data
#     binned_spectrums, tanimoto_scores_df, ms2ds_binner = create_test_data()
#
#     # Define other parameters
#     batch_size = 4
#     dimension = 88
#
#     # Create normal generator
#     normal_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
#                                                  reference_scores_df=tanimoto_scores_df,
#                                                  spectrum_binner=ms2ds_binner, batch_size=batch_size,
#                                                  use_fixed_set=False)
#
#     # Create generator that generates a fixed set every epoch
#     fixed_generator = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
#                                                 reference_scores_df=tanimoto_scores_df,
#                                                 spectrum_binner=ms2ds_binner, batch_size=batch_size,
#                                                 num_turns=5, use_fixed_set=True, random_seed=0)
#
#     first_X, first_y = collect_results(normal_generator, batch_size, dimension)
#     second_X, second_y = collect_results(normal_generator, batch_size, dimension)
#     assert not np.array_equal(first_X, second_X)
#     assert first_y.shape == (4, 2), "Expected different number of labels"
#
#     first_X, first_y = collect_results(fixed_generator, batch_size, dimension)
#     second_X, second_y = collect_results(fixed_generator, batch_size, dimension)
#     assert np.array_equal(first_X, second_X)
#     assert first_y.shape == (4, 10), "Expected different number of labels"
#
#     # Create another fixed generator based on the same dataset that should generate the same
#     # fixed set
#     fixed_generator2 = DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums[:8],
#                                                  reference_scores_df=tanimoto_scores_df,
#                                                  spectrum_binner=ms2ds_binner, batch_size=batch_size,
#                                                  num_turns=5, use_fixed_set=True,
#                                                  random_seed=0)
#     first_X, first_y = collect_results(fixed_generator, batch_size, dimension)
#     second_X, second_y = collect_results(fixed_generator2, batch_size, dimension)
#
#     assert np.array_equal(first_X, second_X)
#
#
# def test_DataGeneratorAllSpectrums_additional_inputs():
#     """
#     Test if additional input parameter works as intended
#     """
#     # Get test data
#     spectrums = load_processed_spectrums()
#     tanimoto_scores_df = get_reference_scores()
#
#     # Run for two test cases.
#     # Testing a single and multiple inputs is important, since numpy can do weird things with 1D arrays of len= 1
#     test_cases = [(StandardScaler("precursor_mz", mean=0, std=1000), ),
#                   (StandardScaler("precursor_mz", mean=0, std=1000),
#                    CategoricalToBinary("ionmode", entries_becoming_one="negative", entries_becoming_zero="positive"))]
#     for additional_feature_types in test_cases:
#
#         # additional_feature_types = ()
#         ms2ds_binner = SpectrumBinner(100, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
#                                       additional_metadata=additional_feature_types)
#         binned_spectrums = ms2ds_binner.fit_transform(spectrums)
#
#         # Define other parameters
#         batch_size = 4
#         data_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
#                                                    spectrum_binner=ms2ds_binner, batch_size=batch_size)
#         batch_X, batch_y = data_generator.__getitem__(0)
#         for batch_X_values in batch_X:
#             assert len(batch_X_values) == len(batch_y) == batch_size, "Batchsizes from X and y are not the same."
#         assert len(batch_X[1][0]) == len(additional_feature_types) == len(batch_X[3][0]), "There are not as many inputs as specified."

