import numpy as np
import torch
from ms2deepscore.models.loss_functions import (risk_aware_mae, risk_aware_mse,
                                                rmse_loss, RiskAwareMAE, bin_dependent_losses)


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


def test_risk_aware_mae_class():
    """Just test if class and function are consistent."""
    model = RiskAwareMAE()
    outputs = torch.tensor([0.2, 0.4, 0.6])
    targets = torch.tensor([0.3, 0.5, 0.7])
    assert model(outputs, targets) == risk_aware_mae(outputs, targets), "Class-based Risk-aware MAE mismatch"


def test_bin_dependent_losses():
    predictions = np.array([0.2, 0.4, 0.6, 0.8])
    true_values = np.array([0.3, 0.5, 0.7, 0.9])
    ref_score_bins = [(0.0, 0.5), (0.5, 1.0)]
    loss_types = ["mse", "mae"]
    bin_content, _, losses = bin_dependent_losses(predictions, true_values, ref_score_bins, loss_types)
    assert len(bin_content) == 2, "Bin content length mismatch"
    assert len(losses["mse"]) == 2, "Losses for MSE mismatch"
    assert len(losses["mae"]) == 2, "Losses for MAE mismatch"
