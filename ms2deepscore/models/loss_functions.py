import numpy as np
import torch
from torch import nn


def weighted_linear(target, weighting_factor):
    """Calculate weight based on target and parameter weighting_factor.
    """
    if weighting_factor < -1 or weighting_factor > 1:
        raise ValueError("For MS2DeepScore models, the weighting_factor should be in the range of -1 to 1.")
    return (1 - weighting_factor) + weighting_factor * target


def mae_loss(outputs, targets, weighting_factor=0):
    weights = weighted_linear(targets, weighting_factor)
    weighted_errors = weights * (torch.abs(outputs - targets))
    return torch.mean(weighted_errors)


def mse_loss(outputs, targets, weighting_factor=0):
    weights = weighted_linear(targets, weighting_factor)
    weighted_errors = weights * ((outputs - targets) ** 2)
    return torch.mean(weighted_errors)


def rmse_loss(outputs, targets, weighting_factor=0):
    return torch.sqrt(mse_loss(outputs, targets, weighting_factor))


def risk_aware_mae(outputs, targets, weighting_factor=0):
    """MAE weighted by target position on scale 0 to 1.
    """
    weights = weighted_linear(targets, weighting_factor)
    errors = weights * (targets - outputs)
    uppers = targets * errors
    lowers = (targets - 1) * errors
    losses = torch.max(lowers, uppers)
    return losses.mean()


def risk_aware_mse(outputs, targets, weighting_factor=0):
    """MSE weighted by target position on scale 0 to 1.
    """
    weights = weighted_linear(targets, weighting_factor)
    errors = weights * (targets - outputs)
    errors = torch.sign(errors) * errors ** 2
    uppers = targets * errors
    lowers = (targets - 1) * errors
    losses = torch.max(lowers, uppers)
    return losses.mean()


class RiskAwareMAE(nn.Module):
    """Loss functions taking into account the actual distribution of the target labels"""
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
    "mse": mse_loss,
    "mae": mae_loss,
    "rmse": rmse_loss,
    "risk_mae": risk_aware_mae,
    "risk_mse": risk_aware_mse,
}
