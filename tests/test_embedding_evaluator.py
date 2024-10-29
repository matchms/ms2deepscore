import os
import pytest
import numpy as np
from sklearn.datasets import make_regression
import torch
from ms2deepscore.models import EmbeddingEvaluationModel, LinearModel
from ms2deepscore.models import load_linear_model, load_embedding_evaluator
from tests.test_data_generators import data_generator_embedding_evaluation, MockMS2DSModel
from tests.create_test_spectra import create_test_spectra
from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator


# This is just to make the ruff linter happy
fixtures = [data_generator_embedding_evaluation]


@pytest.fixture
def mock_settings():
    return SettingsEmbeddingEvaluator(evaluator_num_filters=32,
                                      evaluator_depth=6,
                                      evaluator_kernel_size=40,
                                      mini_batch_size=10,
                                      batches_per_iteration=5,
                                      learning_rate=0.001,
                                      num_epochs=1,
                                      evaluator_distribution_size=10)


@pytest.fixture
def embedding_model(mock_settings):
    return EmbeddingEvaluationModel(settings=mock_settings)


def test_model_initialization(embedding_model):
    """
    Test if the model initializes with the correct number of filters, depth, and kernel size.
    """
    assert embedding_model.settings.evaluator_num_filters == 32, "Incorrect number of filters"
    assert embedding_model.settings.evaluator_depth == 6, "Incorrect depth"
    assert embedding_model.settings.evaluator_kernel_size == 40, "Incorrect kernel size"


def test_forward_pass(embedding_model):
    """
    Test the forward pass of the model with a mock input.
    """
    mock_input = torch.randn(1, 1, 500)
    output = embedding_model(mock_input)
    assert output.shape == (1, 1), "Output shape is incorrect"


def test_model_with_different_input_sizes(embedding_model):
    """
    Test the model with different input sizes to ensure it can handle variable sequence lengths.
    """
    sizes = [100, 250, 500, 750]
    for size in sizes:
        mock_input = torch.randn(1, 1, size)
        output = embedding_model(mock_input)
        assert output.shape == (1, 1), f"Output shape is incorrect for input size {size}"


def test_model_with_batch_sizes(embedding_model):
    """
    Test the model with different input sizes to ensure it can handle variable sequence lengths.
    """
    batch_sizes = [1, 2, 10]
    for size in batch_sizes:
        mock_input = torch.randn(size, 1, 100)
        output = embedding_model(mock_input)
        assert output.shape == (size, 1), f"Output shape is incorrect for batch size {size}"


def test_model_save_load(tmp_path, embedding_model):
    # Save the model
    filepath = tmp_path / "embedding_model.pth"
    embedding_model.save(filepath)

    # Load the model
    loaded_model = load_embedding_evaluator(filepath)

    # Verify if the saved settings and state dict match the original model
    assert loaded_model.settings.evaluator_num_filters == embedding_model.settings.evaluator_num_filters
    assert loaded_model.state_dict().keys() == embedding_model.state_dict().keys()


def test_train_embedding_evaluator(embedding_model, data_generator_embedding_evaluation):
    embedding_model.train_evaluator(create_test_spectra(25), MockMS2DSModel())
    embedding = data_generator_embedding_evaluation.__next__()[2]
    result = embedding_model.compute_embedding_evaluations(embedding)
    assert result.shape == (10, 1)


def test_linear_model_fit_predict():
    # Generate a simple regression problem
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    # Initialize and fit the model
    model = LinearModel(degree=2)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Check predictions shape
    assert predictions.shape == y.shape, "Prediction shape mismatch."


def test_linear_model_save_load(tmp_path):
    temp_filepath = os.path.join(tmp_path, "temp_model.json")

    # Generate a simple regression problem
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    # Initialize and fit the model
    model = LinearModel(degree=3)
    model.fit(X, y)

    # Save the model
    model.save(temp_filepath)

    # Ensure the file was created
    assert os.path.exists(temp_filepath), "Model file was not created."

    # Load the model
    loaded_model = load_linear_model(temp_filepath)

    # Verify the loaded model's parameters match the original model's parameters
    assert np.array_equal(model.model.coef_, loaded_model.model.coef_), "Coefficients do not match."
    assert model.model.intercept_ == loaded_model.model.intercept_, "Intercepts do not match."
    assert model.degree == loaded_model.degree == 3, "Degree does not match."
