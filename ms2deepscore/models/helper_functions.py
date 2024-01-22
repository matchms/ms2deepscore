"""Custom loss and helper function."""
import torch
from torch import nn


def rmse_loss(outputs, targets):
    return torch.sqrt(torch.mean((outputs - targets) ** 2))


def l1_regularization(model, lambda_l1):
    """L1 regulatization for first dense layer of model."""
    l1_loss = torch.linalg.vector_norm(next(model.encoder.dense_layers[0].parameters()), ord=1)
    return lambda_l1 * l1_loss


def l2_regularization(model, lambda_l2):
    """L2 regulatization for first dense layer of model."""
    l2_loss = torch.linalg.vector_norm(next(model.encoder.dense_layers[0].parameters()), ord=2)
    return lambda_l2 * l2_loss


def risk_aware_mae(outputs, targets):
    """MAE weighted by target position on scale 0 to 1.
    """
    factors = targets  # this is meant for a uniform distribution of targets between 0 and 1.

    errors = targets - outputs
    uppers =  factors * errors
    lowers = (factors - 1) * errors

    losses = torch.max(lowers, uppers)
    return losses.mean()


def risk_aware_mse(outputs, targets):
    """MSE weighted by target position on scale 0 to 1.
    """
    factors = targets  # this is meant for a uniform distribution of targets between 0 and 1.

    errors = targets - outputs
    errors = torch.sign(errors) * errors ** 2
    uppers =  factors * errors
    lowers = (factors - 1) * errors

    losses = torch.max(lowers, uppers)
    return losses.mean()


### Loss functions taking into account the actual distribution of the target labels

class RiskAwareMAE(nn.Module):
    def __init__(self, percentiles=None, device="cpu"):
        super().__init__()
        self.device = device
        if percentiles is None:
            self.percentiles = torch.linspace(0.01, 1.0, 100)
        else:
            self.percentiles = percentiles

    def forward(self, outputs, targets):
        device = self.device
        idx = torch.empty((len(targets)))
        for i, target in enumerate(targets):
            idx[i] = torch.argmin(torch.abs(self.percentiles.to(device) - target.to(device)))

        max_bin = self.percentiles.shape[0]
        factors = (idx + 1) / max_bin
    
        errors = targets.to(device) - outputs.to(device)
        uppers =  factors.to(device) * errors
        lowers = (factors.to(device) - 1) * errors
    
        losses = torch.max(lowers, uppers)
        return losses.mean()


LOSS_FUNCTIONS = {
    "mse": nn.MSELoss(),
    "rmse": rmse_loss,
    "risk_mae": risk_aware_mae,
    "risk_mse": risk_aware_mse,
}
