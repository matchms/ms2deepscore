import re

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression

from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator
from ms2deepscore.models import (
    EmbeddingEvaluationModel,
    LinearModel,
    load_embedding_evaluator,
    load_linear_model,
)
from tests.create_test_spectra import create_test_spectra


@pytest.fixture
def evaluator_settings():
    # Keep the test model intentionally small: architecture semantics are the same,
    # while forward/training tests remain fast.
    return SettingsEmbeddingEvaluator(
        evaluator_num_filters=8,
        evaluator_depth=3,
        evaluator_kernel_size=15,
        mini_batch_size=5,
        batches_per_iteration=1,
        learning_rate=0.001,
        num_epochs=1,
        evaluator_distribution_size=10,
        random_seed=13,
    )


@pytest.fixture
def embedding_model(evaluator_settings):
    return EmbeddingEvaluationModel(settings=evaluator_settings)


def test_model_initialization_matches_settings(embedding_model, evaluator_settings):
    assert embedding_model.settings.get_dict() == evaluator_settings.get_dict()
    assert len(embedding_model.inception_block.inception_modules) == evaluator_settings.evaluator_depth
    assert embedding_model.fc.in_features == evaluator_settings.evaluator_num_filters * 4
    assert embedding_model.fc.out_features == 1


@pytest.mark.parametrize(
    ("batch_size", "embedding_size"),
    [(1, 100), (2, 250), (10, 500), (3, 750)],
)
def test_forward_pass_supports_batch_and_embedding_sizes(
    embedding_model,
    batch_size,
    embedding_size,
):
    embedding_model.eval()
    mock_input = torch.randn(batch_size, 1, embedding_size)

    with torch.no_grad():
        output = embedding_model(mock_input)

    assert output.shape == (batch_size, 1)
    assert torch.isfinite(output).all()


def test_compute_embedding_evaluations_returns_flat_numpy_array(embedding_model):
    embedding_model.eval()
    embeddings = np.random.default_rng(4).normal(size=(7, 128)).astype(np.float32)

    result = embedding_model.compute_embedding_evaluations(embeddings, device="cpu")

    assert isinstance(result, np.ndarray)
    assert result.shape == (7,)
    assert np.isfinite(result).all()


def test_model_save_load_roundtrip_preserves_settings_weights_and_predictions(
    tmp_path,
    embedding_model,
):
    filepath = tmp_path / "embedding_model.pt"
    embedding_model.eval()
    test_input = torch.randn(4, 1, 128)

    with torch.no_grad():
        expected_predictions = embedding_model(test_input).clone()

    embedding_model.save(filepath)
    loaded_model = load_embedding_evaluator(filepath)

    assert loaded_model.settings.get_dict() == embedding_model.settings.get_dict()
    assert loaded_model.state_dict().keys() == embedding_model.state_dict().keys()
    for name, expected in embedding_model.state_dict().items():
        torch.testing.assert_close(loaded_model.state_dict()[name], expected)

    with torch.no_grad():
        actual_predictions = loaded_model(test_input)
    torch.testing.assert_close(actual_predictions, expected_predictions)


def test_train_embedding_evaluator_updates_model_parameters(
    monkeypatch,
    embedding_model,
    mock_ms2ds_model,
):
    class FakeDataGenerator:
        def __init__(self, spectrums, ms2ds_model, settings, device="cpu"):
            self.batch_size = settings.evaluator_distribution_size
            self.embedding_dim = ms2ds_model.model_settings.embedding_dim

        def __iter__(self):
            # Zero target MSE, deterministic non-trivial embeddings.
            tanimoto_scores = torch.zeros((self.batch_size, self.batch_size))
            ms2ds_scores = torch.zeros_like(tanimoto_scores)
            embeddings = torch.linspace(
                0.0,
                1.0,
                self.batch_size * self.embedding_dim,
            ).reshape(self.batch_size, self.embedding_dim)
            yield tanimoto_scores, ms2ds_scores, embeddings

    monkeypatch.setattr(
        "ms2deepscore.models.EmbeddingEvaluatorModel.DataGeneratorEmbeddingEvaluation",
        FakeDataGenerator,
    )
    monkeypatch.setattr(
        "ms2deepscore.models.EmbeddingEvaluatorModel.initialize_device",
        lambda: torch.device("cpu"),
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in embedding_model.named_parameters()
    }

    embedding_model.train_evaluator(
        training_spectra=create_test_spectra(2),
        ms2ds_model=mock_ms2ds_model,
    )

    changed = [
        not torch.equal(before[name], parameter.detach())
        for name, parameter in embedding_model.named_parameters()
    ]
    assert any(changed), "Training completed without updating any model parameter."


def test_train_embedding_evaluator_reports_actual_validation_loss(
    monkeypatch,
    capsys,
    embedding_model,
    mock_ms2ds_model,
):
    """Regression test for accidentally reporting the last training loss as val loss."""

    class FakeDataGenerator:
        instance_count = 0

        def __init__(self, spectrums, ms2ds_model, settings, device="cpu"):
            self.batch_size = settings.evaluator_distribution_size
            self.embedding_dim = ms2ds_model.model_settings.embedding_dim
            self.is_validation = FakeDataGenerator.instance_count == 1
            FakeDataGenerator.instance_count += 1

        def __iter__(self):
            embeddings = torch.linspace(
                0.0,
                1.0,
                self.batch_size * self.embedding_dim,
            ).reshape(self.batch_size, self.embedding_dim)
            ms2ds_scores = torch.zeros((self.batch_size, self.batch_size))
            if self.is_validation:
                # Deliberately make the validation target very different from training.
                tanimoto_scores = torch.full_like(ms2ds_scores, 10.0)
            else:
                tanimoto_scores = torch.zeros_like(ms2ds_scores)
            yield tanimoto_scores, ms2ds_scores, embeddings

    class RecordingMSELoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = []

        def forward(self, outputs, targets):
            loss = ((outputs - targets) ** 2).mean()
            self.values.append(float(loss.detach()))
            return loss

    criterion = RecordingMSELoss()
    monkeypatch.setattr(
        "ms2deepscore.models.EmbeddingEvaluatorModel.DataGeneratorEmbeddingEvaluation",
        FakeDataGenerator,
    )
    monkeypatch.setattr(
        "ms2deepscore.models.EmbeddingEvaluatorModel.initialize_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "ms2deepscore.models.EmbeddingEvaluatorModel.nn.MSELoss",
        lambda: criterion,
    )

    training_spectra = create_test_spectra(2)
    validation_spectra = create_test_spectra(2)
    embedding_model.train_evaluator(
        training_spectra=training_spectra,
        validation_spectra=validation_spectra,
        ms2ds_model=mock_ms2ds_model,
    )

    stdout = capsys.readouterr().out
    match = re.search(r"Val_loss:\s*([0-9.eE+-]+)", stdout)
    assert match is not None, "Expected validation loss to be reported."
    reported_validation_loss = float(match.group(1))

    # The final criterion call is the validation batch. This catches code such as
    # `val_losses.append(loss_value)` where loss_value still refers to training.
    expected_validation_loss = criterion.values[-1]
    assert reported_validation_loss == pytest.approx(expected_validation_loss, rel=1e-5, abs=1e-6)

    # train_evaluator should restore training mode after temporary validation eval mode.
    assert embedding_model.training


def test_linear_model_fit_predict():
    X, y = make_regression(
        n_samples=100,
        n_features=2,
        noise=0.1,
        random_state=7,
    )
    model = LinearModel(degree=2)

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()


def test_linear_model_save_load_roundtrip_preserves_predictions(tmp_path):
    X, y = make_regression(
        n_samples=100,
        n_features=2,
        noise=0.1,
        random_state=11,
    )
    model = LinearModel(degree=3)
    model.fit(X, y)
    expected = model.predict(X)

    filepath = tmp_path / "linear_model.json"
    model.save(filepath)
    loaded_model = load_linear_model(filepath)
    actual = loaded_model.predict(X)

    assert filepath.is_file()
    assert loaded_model.degree == model.degree == 3
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
