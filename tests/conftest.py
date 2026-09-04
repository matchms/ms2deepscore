import matplotlib
import numpy as np
import pytest
import torch
from torch import nn

from ms2deepscore.SettingsMS2Deepscore import (
    SettingsEmbeddingEvaluator,
    SettingsMS2Deepscore,
)
from ms2deepscore.train_new_model.DataGeneratorEmbeddingEvaluation import (
    DataGeneratorEmbeddingEvaluation,
)
from tests.create_test_spectra import create_test_spectra
matplotlib.use("Agg", force=True)


class DeterministicMockEncoder(nn.Module):
    """Small deterministic encoder implementing the interface used by the generator.

    Besides returning stable embeddings, it records whether inference happened in
    eval mode and with gradients disabled. This makes the mock useful for testing
    inference semantics without constructing a full SiameseSpectralModel.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.last_training = None
        self.last_grad_enabled = None

    def forward(self, spec_tensors, meta_tensors):
        self.last_training = self.training
        self.last_grad_enabled = torch.is_grad_enabled()

        # Produce deterministic, non-zero embeddings for every input spectrum.
        # The weighted spectral sum makes embeddings spectrum-dependent, while the
        # offsets prevent zero vectors (important for cosine similarities).
        weights = torch.linspace(
            0.5,
            1.5,
            spec_tensors.shape[1],
            dtype=spec_tensors.dtype,
            device=spec_tensors.device,
        )
        summary = (spec_tensors * weights).sum(dim=1, keepdim=True)
        if meta_tensors.shape[1] > 0:
            summary = summary + meta_tensors.sum(dim=1, keepdim=True)

        offsets = torch.linspace(
            0.01,
            1.0,
            self.embedding_dim,
            dtype=spec_tensors.dtype,
            device=spec_tensors.device,
        ).unsqueeze(0)
        return summary + offsets


class MockMS2DSModel(nn.Module):
    """Minimal, valid nn.Module satisfying DataGeneratorEmbeddingEvaluation."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.model_settings = SettingsMS2Deepscore(
            embedding_dim=embedding_dim,
            fingerprint_nbits=128,
        )
        self.encoder = DeterministicMockEncoder(embedding_dim=embedding_dim)


@pytest.fixture
def mock_ms2ds_model():
    return MockMS2DSModel()


@pytest.fixture
def embedding_evaluator_generator_settings():
    return SettingsEmbeddingEvaluator(
        evaluator_distribution_size=10,
        random_seed=123,
    )


@pytest.fixture
def data_generator_embedding_evaluation(
    mock_ms2ds_model,
    embedding_evaluator_generator_settings,
):
    spectra = create_test_spectra(
        num_of_unique_inchikeys=25,
        num_of_spectra_per_inchikey=2,
    )
    return DataGeneratorEmbeddingEvaluation(
        spectrums=spectra,
        ms2ds_model=mock_ms2ds_model,
        settings=embedding_evaluator_generator_settings,
        device="cpu",
    )
