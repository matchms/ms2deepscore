import string
import pytest
import numpy as np
import torch
from matchms import Spectrum
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore, SettingsEmbeddingEvaluator
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.data_generators import DataGeneratorPytorch,\
    DataGeneratorEmbeddingEvaluation
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper


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
    spectrums = create_test_spectra(num_of_unique_inchikeys=25)
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


def test_DataGeneratorPytorch():
    """Test DataGeneratorPytorch using generated data.
    """
    num_of_unique_inchikeys = 15
    spectrums = create_test_spectra(num_of_unique_inchikeys)
    batch_size = 8
    settings = SettingsMS2Deepscore(min_mz=10,
                                    max_mz=1000,
                                    mz_bin_width=0.1,
                                    intensity_scaling=0.5,
                                    additional_metadata=[],
                                    same_prob_bins=np.array([(x / 4, x / 4 + 0.25) for x in range(0, 4)]),
                                    average_pairs_per_bin=1,
                                    batch_size=batch_size,
                                    augment_removal_max=0.0,
                                    augment_removal_intensity=0.0,
                                    augment_intensity=0.0,
                                    augment_noise_max=0)
    scp, spectrums = select_compound_pairs_wrapper(spectrums, settings)

    # Create generator
    test_generator = DataGeneratorPytorch(spectrums=spectrums, selected_compound_pairs=scp, settings=settings)

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
    assert tanimoto_scores.shape == (data_generator_embedding_evaluation.batch_size, data_generator_embedding_evaluation.batch_size), "Incorrect shape for tanimoto_scores"
    assert ms2ds_scores.shape == (data_generator_embedding_evaluation.batch_size, data_generator_embedding_evaluation.batch_size), "Incorrect shape for ms2ds_scores"
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
    assert not np.array_equal(data_generator_embedding_evaluation.indexes, initial_indexes), "Indexes not shuffled after epoch end"
