import torch
from ms2deepscore.models.loss_functions import risk_aware_mae, risk_aware_mse


def test_risk_aware_mae():
    assert risk_aware_mae(torch.tensor([0.7]), torch.tensor([0.7]))  == 0
    assert risk_aware_mae(torch.tensor([0.7]), torch.tensor([0.5])) == risk_aware_mae(torch.tensor([0.3]), torch.tensor([0.5]))
    assert torch.allclose(risk_aware_mae(torch.tensor([0.3]), torch.tensor([0.5])), torch.tensor(0.1))
    assert torch.allclose(risk_aware_mae(torch.tensor([0.4]), torch.tensor([0.8])), torch.tensor(0.32))


def test_risk_aware_mse():
    assert risk_aware_mse(torch.tensor([0.7]), torch.tensor([0.7]))  == 0
    assert risk_aware_mse(torch.tensor([0.7]), torch.tensor([0.5])) == risk_aware_mse(torch.tensor([0.3]), torch.tensor([0.5]))
    assert torch.allclose(risk_aware_mse(torch.tensor([0.3]), torch.tensor([0.5])), torch.tensor(0.2 ** 2 * 0.5))
    assert torch.allclose(risk_aware_mse(torch.tensor([0.4]), torch.tensor([0.8])), torch.tensor(0.4 ** 2 * 0.8))