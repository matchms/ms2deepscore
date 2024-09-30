import pytest
import numpy as np
import torch
from collections import Counter
from matchms import Spectrum
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore, SettingsEmbeddingEvaluator
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.data_generators import SpectrumPairGenerator, \
    DataGeneratorEmbeddingEvaluation, create_data_generator
from ms2deepscore.train_new_model import InchikeyPairGenerator
from tests.create_test_spectra import create_test_spectra


class MockMS2DSModel:
    def __init__(self):
        self.model_settings = SettingsMS2Deepscore()

    def encoder(self, spec_tensors, meta_tensors):
        # Return mock embeddings as random tensors
        return torch.rand(spec_tensors.size(0), 128)  # Assuming embedding size of 128

    def to(self, device):
        pass


@pytest.fixture
def data_generator_embedding_evaluation():
    spectrums = create_test_spectra(num_of_unique_inchikeys=25, num_of_spectra_per_inchikey=2)
    params = {"evaluator_distribution_size": 10}
    return DataGeneratorEmbeddingEvaluation(spectrums=spectrums,
                                            ms2ds_model=MockMS2DSModel(),
                                            settings=SettingsEmbeddingEvaluator(**params),
                                            device="cpu")


def collect_results(generator, batch_size, dimension):
    n_batches = len(generator)
    X = np.zeros((batch_size, dimension, 2, n_batches))
    y = np.zeros((batch_size, n_batches))
    for i, batch in enumerate(generator):
        X[:, :, 0, i] = batch[0][0]
        X[:, :, 1, i] = batch[0][1]
        y[:, i] = batch[1]
    return X, y


def test_tensorize_spectra():
    spectrum = Spectrum(mz=np.array([10, 500, 999.9]), intensities=np.array([0.5, 0.5, 1]))
    settings = SettingsMS2Deepscore(min_mz=10,
                                    max_mz=1000,
                                    mz_bin_width=1.0,
                                    intensity_scaling=0.5,
                                    additional_metadata=[])
    spec_tensors, meta_tensors = tensorize_spectra([spectrum, spectrum], settings)

    assert meta_tensors.shape == torch.Size([2, 0])
    assert spec_tensors.shape == torch.Size([2, 990])
    assert spec_tensors[0, 0] == spec_tensors[0, 490] == 0.5 ** 0.5
    assert spec_tensors[0, -1] == 1


@pytest.fixture()
def dummy_data_generator():
    spectrums = create_test_spectra(4, 3)
    selected_pairs = InchikeyPairGenerator([('CCCCCCCCCCCCCC', 'DDDDDDDDDDDDDD', 0.25),
                                            ('BBBBBBBBBBBBBB', 'DDDDDDDDDDDDDD', 0.6666667),
                                            ('AAAAAAAAAAAAAA', 'CCCCCCCCCCCCCC', 1.0),
                                            ('AAAAAAAAAAAAAA', 'BBBBBBBBBBBBBB', 0.33333334)])
    batch_size = 2
    settings = SettingsMS2Deepscore(min_mz=10,
                                    max_mz=1000,
                                    mz_bin_width=0.1,
                                    intensity_scaling=0.5,
                                    additional_metadata=[],
                                    same_prob_bins=np.array([(-0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]),
                                    batch_size=batch_size,
                                    num_turns=4,
                                    augment_removal_max=0.0,
                                    augment_removal_intensity=0.0,
                                    augment_intensity=0.0,
                                    augment_noise_max=0)
    return SpectrumPairGenerator(spectrums, selected_pairs, settings)


def test_correct_batch_format_data_generator(dummy_data_generator):
    def check_correct_batch_format(batch, batch_size=2):
        """Checks that each output has the shape of the batch and the expected tensor shapes"""
        assert len(batch) == 5, "expected 5 tensors as output"
        for i in range(5):
            assert batch[i].shape[0] == batch_size
        spec1, spec2, meta1, meta2, targets = batch
        assert meta1.shape[1] == meta2.shape[1] == 0
        assert spec1.shape[1] == spec2.shape[1] == 9900
        assert targets.shape[0] == batch_size

    batch = dummy_data_generator.__getitem__(0)
    check_correct_batch_format(batch)

    assert len(dummy_data_generator) == 8

    for batch in dummy_data_generator:
        check_correct_batch_format(batch)


def test_equal_sampling_of_spectra(dummy_data_generator):
    """Tests that all unique spectra are at least sampled once.
    The sampling is random, but for enough repetitions very likely to always happen.
    This test is mostly to make sure we don't accidentally implement something
    where we just resample the same spectrum every time for one inchikey"""
    tensorized_spectra = []
    epochs = 20
    for _ in range(epochs):
        for batch in dummy_data_generator:
            for i in range(batch[0].shape[0]):
                tensorized_spectra.append(tuple(batch[0][i].tolist()))
                tensorized_spectra.append(tuple(batch[1][i].tolist()))
    # Count occurrences of each unique tensor, the dummy spectra are generated, so they all result in unique tensors.
    tensor_counts = {}
    for spectrum_tensor in tensorized_spectra:
        if spectrum_tensor in tensor_counts:
            tensor_counts[spectrum_tensor] += 1
        else:
            tensor_counts[spectrum_tensor] = 1
    # test if all spectra are sampled (at least once)
    unique_tensors = tensor_counts.keys()
    # Test that each spectrum is sampled. This is not really always true, since we randomly sample spectra,
    # but since we sample 640 spectra from 24 options, it is very unlikely (1 in 28 billion)
    # that this will result in not sampling all at least once.
    # Because we have a fixed seed, this should not result in random failing tests.
    assert len(unique_tensors) == len(dummy_data_generator.spectrums), "Not all spectra are selected at least once"

    def reverse_tensorize(tensor, list_of_spectra, settings):
        """Finds the spectrum in a list of spectra based on the tensorized vesion"""
        # Create tensors of the available spectra, to later make it possible to link spectra back to inchikeys again.
        tensorized_spectra, _ = tensorize_spectra(list_of_spectra, settings)
        list_of_spectrum_tensors = [tuple(tensor.tolist()) for tensor in tensorized_spectra]
        assert len(set(list_of_spectrum_tensors)) == len(list_of_spectrum_tensors), \
            "There are repeating tensors, meaning that there are spectra that result in exactly the same tensor. " \
            "Change the dummy spectra to have unique spectra."
        for i, tensorized_spectrum in enumerate(list_of_spectrum_tensors):
            if tensorized_spectrum == tensor:
                return list_of_spectra[i]

    # get spectrum counts per inchikey (by reverse engineering which tensors belong to which spectrum)
    inchikey_counts = Counter()
    for unique_tensor, count in tensor_counts.items():
        spectrum = reverse_tensorize(unique_tensor,
                                     dummy_data_generator.spectrums,
                                     dummy_data_generator.model_settings)

        inchikey = spectrum.get("inchikey")[:14]
        inchikey_counts[inchikey] += count
    # Test that the inchikeys are sampled equally
    assert max(inchikey_counts.values()) - min(inchikey_counts.values()) < 2


def test_create_data_generator():
    """tests if a the function create_data_generator creates a datagenerator that samples all input spectra
    correct distributions of inchikeys and scores are tested in other tests"""
    test_spectra = create_test_spectra(8, 3)
    data_generator = create_data_generator(training_spectra=test_spectra,
                                           settings=SettingsMS2Deepscore(
                                               min_mz=10,
                                               max_mz=1000,
                                               mz_bin_width=0.1,
                                               intensity_scaling=0.5,
                                               additional_metadata=[],
                                               same_prob_bins=np.array([(-0.000001, 0.25), (0.25, 0.5), (0.5, 0.75),
                                                                        (0.75, 1)]),
                                               batch_size=2,
                                               num_turns=4,
                                               augment_removal_max=0.0,
                                               augment_removal_intensity=0.0,
                                               augment_intensity=0.0,
                                               augment_noise_max=0))
    tensorized_spectra = []
    epochs = 20
    for _ in range(epochs):
        for batch in data_generator:
            for i in range(batch[0].shape[0]):
                tensorized_spectra.append(tuple(batch[0][i].tolist()))
                tensorized_spectra.append(tuple(batch[1][i].tolist()))
    # Count occurrences of each unique tensor, the dummy spectra are generated, so they all result in unique tensors.
    tensor_counts = {}
    for spectrum_tensor in tensorized_spectra:
        if spectrum_tensor in tensor_counts:
            tensor_counts[spectrum_tensor] += 1
        else:
            tensor_counts[spectrum_tensor] = 1
    # test if all spectra are sampled (at least once)
    unique_tensors = tensor_counts.keys()
    # Test that each spectrum is sampled. This is not really always true, since we randomly sample spectra,
    # but since we sample 640 spectra from 24 options, it is very unlikely (1 in 28 billion)
    # that this will result in not sampling all at least once.
    # Because we have a fixed seed, this should not result in random failing tests.
    assert len(unique_tensors) == len(test_spectra), "Not all spectra are selected at least once"


### Tests for EmbeddingEvaluator data generator
def test_generator_initialization(data_generator_embedding_evaluation):
    """
    Test if the data generator initializes correctly.
    """
    assert len(data_generator_embedding_evaluation.spectrums) == 2 * 25, "Incorrect number of spectrums"
    assert data_generator_embedding_evaluation.batch_size == data_generator_embedding_evaluation.settings.evaluator_distribution_size, "Incorrect batch size"


def test_batch_generation(data_generator_embedding_evaluation):
    """
    Test if batches generated are correct in structure and size.
    """
    tanimoto_scores, ms2ds_scores, embeddings = next(data_generator_embedding_evaluation)
    assert tanimoto_scores.shape == (data_generator_embedding_evaluation.batch_size,
                                     data_generator_embedding_evaluation.batch_size), "Incorrect shape for tanimoto_scores"
    assert ms2ds_scores.shape == (data_generator_embedding_evaluation.batch_size,
                                  data_generator_embedding_evaluation.batch_size), "Incorrect shape for ms2ds_scores"
    assert embeddings.shape[0] == data_generator_embedding_evaluation.batch_size, "Incorrect batch size in embeddings"


def test_epoch_end_functionality(data_generator_embedding_evaluation):
    """
    Test if the generator correctly resets and shuffles after an epoch.
    """
    initial_indexes = data_generator_embedding_evaluation.indexes.copy()
    counter = 0
    for _ in data_generator_embedding_evaluation:
        counter += 1
    assert counter == 5
    # 2nd run
    for _ in data_generator_embedding_evaluation:
        counter += 1
    assert counter == 10
    assert not np.array_equal(data_generator_embedding_evaluation.indexes,
                              initial_indexes), "Indexes not shuffled after epoch end"
