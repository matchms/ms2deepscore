import torch
from ms2deepscore.models.loss_functions import (risk_aware_mae, risk_aware_mse,
                                                rmse_loss,
                                                mae_loss, mse_loss)


def test_mae_loss():
    outputs = torch.tensor([3, 4, 5.])
    targets = torch.tensor([2, 3, 6.5])
    assert mae_loss(outputs, outputs) == 0, "Loss should be zero for identical inputs"
    assert mae_loss(outputs, targets) == torch.abs(outputs - targets).mean()


def test_mae_loss_weighted():
    outputs = torch.tensor([0.3, 0.4, 0.5])
    targets = torch.tensor([0.2, 0.35, 0.7])
    assert mae_loss(outputs, outputs, 0.5) == 0, "Loss should be zero for identical inputs"
    weights = 0.5 + 0.5 * targets
    assert mae_loss(outputs, targets, 0.5) == (weights * torch.abs(outputs - targets)).mean()


def test_mse_loss():
    outputs = torch.tensor([3, 4, 5.])
    targets = torch.tensor([2, 3, 6.5])
    assert mse_loss(outputs, outputs) == 0, "MSE Loss should be zero for identical inputs"
    assert mse_loss(outputs, targets) == ((outputs - targets) ** 2).mean()


def test_mse_loss_weighted():
    outputs = torch.tensor([0.3, 0.4, 0.5])
    targets = torch.tensor([0.2, 0.35, 0.7])
    assert mse_loss(outputs, outputs, 0.5) == 0, "MSE Loss should be zero for identical inputs"
    weights = 0.5 + 0.5 * targets
    assert mse_loss(outputs, targets, 0.5) == (weights * (outputs - targets) ** 2).mean()


def test_rmse_loss():
    outputs = torch.tensor([3, 4, 5.])
    targets = torch.tensor([2, 3, 6.5])
    assert rmse_loss(outputs, outputs) == 0, "RMSE Loss should be zero for identical inputs"
    assert rmse_loss(outputs, targets) == (((outputs - targets) ** 2).mean() ** 0.5)


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
