from torch import tensor, abs, allclose
from ms2deepscore.models.loss_functions import risk_aware_mae, risk_aware_mse, rmse_loss, mae_loss, mse_loss


def test_mae_loss():
    outputs = tensor([3, 4, 5.0])
    targets = tensor([2, 3, 6.5])
    assert mae_loss(outputs, outputs) == 0, "Loss should be zero for identical inputs"
    assert mae_loss(outputs, targets) == abs(outputs - targets).mean()


def test_mae_loss_weighted():
    outputs = tensor([0.3, 0.4, 0.5])
    targets = tensor([0.2, 0.35, 0.7])
    assert mae_loss(outputs, outputs, 0.5) == 0, "Loss should be zero for identical inputs"
    weights = 0.5 + 0.5 * targets
    assert mae_loss(outputs, targets, 0.5) == (weights * abs(outputs - targets)).mean()


def test_mse_loss():
    outputs = tensor([3, 4, 5.0])
    targets = tensor([2, 3, 6.5])
    assert mse_loss(outputs, outputs) == 0, "MSE Loss should be zero for identical inputs"
    assert mse_loss(outputs, targets) == ((outputs - targets) ** 2).mean()


def test_mse_loss_weighted():
    outputs = tensor([0.3, 0.4, 0.5])
    targets = tensor([0.2, 0.35, 0.7])
    assert mse_loss(outputs, outputs, 0.5) == 0, "MSE Loss should be zero for identical inputs"
    weights = 0.5 + 0.5 * targets
    assert mse_loss(outputs, targets, 0.5) == (weights * (outputs - targets) ** 2).mean()


def test_rmse_loss():
    outputs = tensor([3, 4, 5.0])
    targets = tensor([2, 3, 6.5])
    assert rmse_loss(outputs, outputs) == 0, "RMSE Loss should be zero for identical inputs"
    assert rmse_loss(outputs, targets) == (((outputs - targets) ** 2).mean() ** 0.5)


def test_risk_aware_mae():
    assert risk_aware_mae(tensor([0.7]), tensor([0.7])) == 0
    assert risk_aware_mae(tensor([0.7]), tensor([0.5])) == risk_aware_mae(tensor([0.3]), tensor([0.5]))
    assert allclose(risk_aware_mae(tensor([0.3]), tensor([0.5])), tensor(0.1))
    assert allclose(risk_aware_mae(tensor([0.4]), tensor([0.8])), tensor(0.32))


def test_risk_aware_mse():
    assert risk_aware_mse(tensor([0.7]), tensor([0.7])) == 0
    assert risk_aware_mse(tensor([0.7]), tensor([0.5])) == risk_aware_mse(tensor([0.3]), tensor([0.5]))
    assert allclose(risk_aware_mse(tensor([0.3]), tensor([0.5])), tensor(0.2**2 * 0.5))
    assert allclose(risk_aware_mse(tensor([0.4]), tensor([0.8])), tensor(0.4**2 * 0.8))
