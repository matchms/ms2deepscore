from collections import Counter

import numpy as np
import pytest
import torch
from matchms import Spectrum

from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator, SettingsMS2Deepscore
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model import (
    SpectrumPairGenerator,
    create_spectrum_pair_generator,
)
from ms2deepscore.train_new_model.DataGeneratorEmbeddingEvaluation import (
    DataGeneratorEmbeddingEvaluation,
)
from ms2deepscore.train_new_model.TrainingBatchGenerator import TrainingBatchGenerator
from ms2deepscore.train_new_model.inchikey_pair_selection_cross_ionmode import (
    create_data_generator_across_ionmodes,
    select_compound_pairs_wrapper_across_ionmode,
)
from tests.create_test_spectra import create_test_spectra


SELECTED_PAIRS = [
    ("CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD", 0.25),
    ("BBBBBBBBBBBBBB", "DDDDDDDDDDDDDD", 0.6666667),
    ("AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC", 1.0),
    ("AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", 0.33333334),
]


def _training_settings(**overrides):
    settings = dict(
        min_mz=10,
        max_mz=1000,
        mz_bin_width=0.1,
        intensity_scaling=0.5,
        additional_metadata=[],
        same_prob_bins=np.array(
            [(-0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        ),
        batch_size=2,
        num_turns=4,
        augment_removal_max=0.0,
        augment_removal_intensity=0.0,
        augment_intensity=0.0,
        augment_noise_max=0,
        average_inchikey_sampling_count=2,
        random_seed=0,
    )
    settings.update(overrides)
    return SettingsMS2Deepscore(**settings)


def _make_training_batch_generator():
    spectra = create_test_spectra(4, 3)
    pair_generator = SpectrumPairGenerator(
        list(SELECTED_PAIRS),
        spectra,
        shuffle=True,
        random_seed=0,
    )
    return TrainingBatchGenerator(pair_generator, _training_settings())


@pytest.fixture
def dummy_data_generator():
    return _make_training_batch_generator()


def _assert_training_batch(batch, batch_size=2, n_bins=9900):
    assert len(batch) == 5
    spec1, spec2, meta1, meta2, targets = batch

    assert spec1.shape == (batch_size, n_bins)
    assert spec2.shape == (batch_size, n_bins)
    assert meta1.shape == (batch_size, 0)
    assert meta2.shape == (batch_size, 0)
    assert targets.shape == (batch_size,)
    assert targets.dtype == torch.float32


def _make_pos_neg_spectra():
    spectra = create_test_spectra(20, 2)
    positive = spectra[:20]
    negative = spectra[20:]
    for spectrum in positive:
        spectrum.set("ionmode", "positive")
    for spectrum in negative:
        spectrum.set("ionmode", "negative")
    return positive, negative


def _cross_ionmode_settings():
    return SettingsMS2Deepscore(
        min_mz=10,
        max_mz=1000,
        mz_bin_width=0.1,
        intensity_scaling=0.5,
        additional_metadata=[],
        same_prob_bins=np.array([(-0.01, 0.6), (0.6, 1.0)]),
        max_inchikey_sampling=300,
        batch_size=2,
        num_turns=4,
        random_seed=11,
    )


# ---------------------------------------------------------------------------
# Spectrum tensorization
# ---------------------------------------------------------------------------


def test_tensorize_spectra_without_metadata():
    spectrum = Spectrum(
        mz=np.array([10.0, 500.0, 999.9]),
        intensities=np.array([0.5, 0.5, 1.0]),
    )
    settings = SettingsMS2Deepscore(
        min_mz=10,
        max_mz=1000,
        mz_bin_width=1.0,
        intensity_scaling=0.5,
        additional_metadata=[],
    )

    spec_tensors, meta_tensors = tensorize_spectra([spectrum, spectrum], settings)

    assert spec_tensors.shape == (2, 990)
    assert meta_tensors.shape == (2, 0)
    torch.testing.assert_close(spec_tensors[0, 0], torch.tensor(0.5**0.5))
    torch.testing.assert_close(spec_tensors[0, 490], torch.tensor(0.5**0.5))
    torch.testing.assert_close(spec_tensors[0, -1], torch.tensor(1.0))


# ---------------------------------------------------------------------------
# TrainingBatchGenerator / SpectrumPairGenerator
# ---------------------------------------------------------------------------


def test_training_batch_generator_batch_contract(dummy_data_generator):
    _assert_training_batch(dummy_data_generator[0])
    assert len(dummy_data_generator) == 8

    batches = list(dummy_data_generator)
    assert len(batches) == 8
    for batch in batches:
        _assert_training_batch(batch)


def test_training_batch_generator_is_reproducible_with_seed():
    generator_1 = _make_training_batch_generator()
    generator_2 = _make_training_batch_generator()

    # Augmentation is disabled in this fixture, so the same pair/spectrum RNG seed
    # should produce identical batches.
    for batch_1, batch_2 in zip(generator_1, generator_2):
        for tensor_1, tensor_2 in zip(batch_1, batch_2):
            torch.testing.assert_close(tensor_1, tensor_2)


def test_spectrum_pair_generator_samples_valid_spectra_and_repeats_indefinitely():
    spectra = create_test_spectra(4, 3)
    generator = SpectrumPairGenerator(
        list(SELECTED_PAIRS),
        spectra,
        shuffle=True,
        random_seed=0,
    )

    valid_inchikeys = {s.get("inchikey")[:14] for s in spectra}
    seen_spectrum_ids = set()

    # Iterate well beyond one pass through selected_inchikey_pairs. This checks both
    # cycling and random selection among multiple spectra belonging to one compound.
    for _ in range(200):
        spectrum_1, spectrum_2, score = next(generator)
        assert spectrum_1.get("inchikey")[:14] in valid_inchikeys
        assert spectrum_2.get("inchikey")[:14] in valid_inchikeys
        assert 0.0 <= float(score) <= 1.0
        seen_spectrum_ids.add(id(spectrum_1))
        seen_spectrum_ids.add(id(spectrum_2))

    # All four compounds occur in SELECTED_PAIRS, each with three spectra. With the
    # fixed RNG seed this is deterministic, while avoiding expensive tensor roundtrips.
    assert len(seen_spectrum_ids) == len(spectra)


def test_spectrum_pair_generator_has_balanced_compound_frequency_for_fixture():
    generator = SpectrumPairGenerator(
        list(SELECTED_PAIRS),
        create_test_spectra(4, 3),
        shuffle=False,
        random_seed=0,
    )

    counts = generator.get_inchikey_counts()

    assert counts == Counter(
        {
            "AAAAAAAAAAAAAA": 2,
            "BBBBBBBBBBBBBB": 2,
            "CCCCCCCCCCCCCC": 2,
            "DDDDDDDDDDDDDD": 2,
        }
    )


def test_create_spectrum_pair_generator_returns_pairs_from_input_compounds():
    spectra = create_test_spectra(8, 3)
    settings = SettingsMS2Deepscore(
        min_mz=10,
        max_mz=1000,
        mz_bin_width=0.1,
        intensity_scaling=0.5,
        additional_metadata=[],
        same_prob_bins=np.array([(-0.01, 0.75), (0.75, 1.0)]),
        batch_size=2,
        num_turns=4,
        augment_removal_max=0.0,
        augment_removal_intensity=0.0,
        augment_intensity=0.0,
        augment_noise_max=0,
        random_seed=7,
    )

    pair_generator = create_spectrum_pair_generator(spectra, settings=settings)

    assert len(pair_generator) > 0
    valid_inchikeys = {s.get("inchikey")[:14] for s in spectra}
    for inchikey_1, inchikey_2, score in pair_generator.selected_inchikey_pairs:
        assert inchikey_1 in valid_inchikeys
        assert inchikey_2 in valid_inchikeys
        assert 0.0 <= float(score) <= 1.0

    # Also ensure the result can actually feed TrainingBatchGenerator.
    batch_generator = TrainingBatchGenerator(pair_generator, settings)
    _assert_training_batch(next(batch_generator))


# ---------------------------------------------------------------------------
# Embedding-evaluator data generator
# ---------------------------------------------------------------------------


def test_embedding_generator_initialization(
    data_generator_embedding_evaluation,
    mock_ms2ds_model,
):
    generator = data_generator_embedding_evaluation

    assert len(generator.spectrums) == 50
    assert generator.batch_size == generator.settings.evaluator_distribution_size == 10
    assert len(generator) == 5

    # The MS2DeepScore model is used only for inference by this generator.
    assert not mock_ms2ds_model.training
    assert not mock_ms2ds_model.encoder.training


def test_embedding_generator_batch_contract_and_inference_mode(
    data_generator_embedding_evaluation,
    mock_ms2ds_model,
):
    tanimoto_scores, ms2ds_scores, embeddings = next(data_generator_embedding_evaluation)
    batch_size = data_generator_embedding_evaluation.batch_size

    assert tanimoto_scores.shape == (batch_size, batch_size)
    assert ms2ds_scores.shape == (batch_size, batch_size)
    assert embeddings.shape == (batch_size, 128)
    assert not embeddings.requires_grad

    # Similarity matrices for a set against itself must be symmetric with unit diagonal.
    np.testing.assert_allclose(tanimoto_scores, tanimoto_scores.T, atol=1e-7)
    np.testing.assert_allclose(ms2ds_scores, ms2ds_scores.T, atol=1e-7)
    np.testing.assert_allclose(np.diag(tanimoto_scores), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.diag(ms2ds_scores), 1.0, atol=1e-6)

    # These assertions explicitly protect the eval()/no_grad() inference contract.
    assert mock_ms2ds_model.encoder.last_training is False
    assert mock_ms2ds_model.encoder.last_grad_enabled is False


def test_embedding_generator_resets_after_each_epoch(data_generator_embedding_evaluation):
    generator = data_generator_embedding_evaluation
    initial_indexes = generator.indexes.copy()

    assert len(list(generator)) == len(generator) == 5
    assert generator.current_index == 0
    indexes_after_epoch_1 = generator.indexes.copy()
    assert not np.array_equal(indexes_after_epoch_1, initial_indexes)

    assert len(list(generator)) == 5
    assert generator.current_index == 0
    indexes_after_epoch_2 = generator.indexes.copy()
    assert not np.array_equal(indexes_after_epoch_2, indexes_after_epoch_1)


def test_embedding_generator_shuffle_is_reproducible_for_same_seed(mock_ms2ds_model):
    settings_1 = SettingsEmbeddingEvaluator(evaluator_distribution_size=10, random_seed=77)
    settings_2 = SettingsEmbeddingEvaluator(evaluator_distribution_size=10, random_seed=77)
    spectra = create_test_spectra(25, 2)

    # Use separate model instances because generator initialization switches them to eval mode.
    model_type = type(mock_ms2ds_model)
    generator_1 = DataGeneratorEmbeddingEvaluation(spectra, model_type(), settings_1, device="cpu")
    generator_2 = DataGeneratorEmbeddingEvaluation(spectra, model_type(), settings_2, device="cpu")

    np.testing.assert_array_equal(generator_1.indexes, generator_2.indexes)
    generator_1.on_epoch_end()
    generator_2.on_epoch_end()
    np.testing.assert_array_equal(generator_1.indexes, generator_2.indexes)


# ---------------------------------------------------------------------------
# Cross-ionmode generators
# ---------------------------------------------------------------------------


def test_create_data_generator_across_ionmodes_returns_valid_batches():
    positive, negative = _make_pos_neg_spectra()
    generator = create_data_generator_across_ionmodes(
        positive + negative,
        _cross_ionmode_settings(),
    )

    assert len(generator) > 0
    # CombinedSpectrumGenerator alternates same-mode and cross-mode pair sources.
    for _ in range(min(len(generator), 6)):
        batch = next(generator)
        _assert_training_batch(batch)
        assert torch.isfinite(batch[-1]).all()


def test_cross_ionmode_pair_generator_preserves_ionmode_direction():
    positive, negative = _make_pos_neg_spectra()
    pair_generator = select_compound_pairs_wrapper_across_ionmode(
        positive,
        negative,
        _cross_ionmode_settings(),
    )

    assert len(pair_generator) > 0
    for _ in range(len(pair_generator)):
        spectrum_1, spectrum_2, score = next(pair_generator)
        assert spectrum_1.get("ionmode") == "positive"
        assert spectrum_2.get("ionmode") == "negative"
        assert 0.0 <= float(score) <= 1.0

    # It is intentionally cyclic/infinite rather than exhausted after one schedule.
    spectrum_1, spectrum_2, _ = next(pair_generator)
    assert spectrum_1.get("ionmode") == "positive"
    assert spectrum_2.get("ionmode") == "negative"
