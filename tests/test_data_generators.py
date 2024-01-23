import os
import string
import numpy as np
import torch
from matchms import Spectrum
from ms2deepscore.train_new_model.data_generators import (DataGeneratorPytorch,
                                                          compute_validation_set,
                                                          tensorize_spectra,
                                                          write_to_pickle,
                                                          load_generator_from_pickle)
from ms2deepscore.SettingsMS2Deepscore import \
    GeneratorSettings, TensorizationSettings
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper


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
    batch_1 = val_generator.__getitem__(0)
    for i, tensor in enumerate(batch_0):
        torch.equal(tensor, batch_0_saved[i])
        torch.equal(tensor, batch_1[i]) # Check if each epoch is the same
    assert "spectrums" not in val_generator.__dict__, "Spectrums should have been removed"
    assert len(val_generator) == 3
    assert torch.allclose(batch_0[4], torch.tensor([0.5000, 0.2500, 0.4286, 0.4286, 0.1429]), atol=1e8)
