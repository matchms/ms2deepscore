import pytest
import torch
from ms2deepscore.models import EmbeddingEvaluationModel
# from ms2deepscor.SettingsMS2Deepscore import SettingsMS2Deepscore


# Mock SettingsMS2Deepscore to avoid dependencies on the actual implementation
class MockSettingsMS2Deepscore:
    def __init__(self):
        self.evaluator_num_filters = 32
        self.evaluator_depth = 6
        self.evaluator_kernel_size = 40


@pytest.fixture
def mock_settings():
    return MockSettingsMS2Deepscore()


@pytest.fixture
def model(mock_settings):
    return EmbeddingEvaluationModel(settings=mock_settings)


def test_model_initialization(model):
    """
    Test if the model initializes with the correct number of filters, depth, and kernel size.
    """
    assert model.settings.evaluator_num_filters == 32, "Incorrect number of filters"
    assert model.settings.evaluator_depth == 6, "Incorrect depth"
    assert model.settings.evaluator_kernel_size == 40, "Incorrect kernel size"


def test_forward_pass(model):
    """
    Test the forward pass of the model with a mock input.
    """
    mock_input = torch.randn(1, 1, 500)
    output = model(mock_input)
    assert output.shape == (1, 1), "Output shape is incorrect"


def test_model_with_different_input_sizes(model):
    """
    Test the model with different input sizes to ensure it can handle variable sequence lengths.
    """
    sizes = [100, 250, 500, 750]
    for size in sizes:
        mock_input = torch.randn(1, 1, size)
        output = model(mock_input)
        assert output.shape == (1, 1), f"Output shape is incorrect for input size {size}"


def test_model_with_batch_sizes(model):
    """
    Test the model with different input sizes to ensure it can handle variable sequence lengths.
    """
    batch_sizes = [1, 2, 10]
    for size in batch_sizes:
        mock_input = torch.randn(size, 1, 100)
        output = model(mock_input)
        assert output.shape == (size, 1), f"Output shape is incorrect for batch size {size}"
