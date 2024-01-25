import numpy as np
import torch
from torch import nn


def rmse_loss(outputs, targets):
    return torch.sqrt(torch.mean((outputs - targets) ** 2))


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
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    "rmse": rmse_loss,
    "risk_mae": risk_aware_mae,
    "risk_mse": risk_aware_mse,
}


def bin_dependent_losses(predictions,
                         true_values,
                         ref_score_bins,
                         loss_types=("mse",),
                         ):
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).

    Parameters
    ----------
    predictions
        Scores that should be evaluated
    true_values
        Reference scores (= ground truth).
    ref_score_bins
        Bins for the reference score to evaluate the performance of scores. in the form [(0.0, 0.1), (0.1, 0.2) ...]
    loss_types
        Specify list of loss types out of "mse", "mae", "rmse", "risk_mae", "risk_mse".
    """
    # pylint: disable=too-many-locals
    if predictions.shape != true_values.shape:
        raise ValueError("Expected true values and predictions to have the same shape")
    bin_content = []
    losses = {"bin": []}
    for loss_type in loss_types:
        if loss_type.lower() not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss function: {loss_type}. Must be one of: {LOSS_FUNCTIONS.keys()}")
        losses[loss_type] = []
    bounds = []
    for i, (low, high) in enumerate(ref_score_bins):
        bounds.append((low, high))
        if i == 0:
            idx = np.where((true_values >= low) & (true_values <= high))
        else:
            idx = np.where((true_values > low) & (true_values <= high))
        if idx[0].shape[0] == 0:
            raise ValueError("No reference scores within bin")
        bin_content.append(idx[0].shape[0])
        # Add values
        losses["bin"].append((low, high))
        for loss_type in loss_types:
            criterion = LOSS_FUNCTIONS[loss_type.lower()]
            selected_true_values = torch.tensor(true_values[idx])
            selected_predictions = torch.tensor(predictions[idx])
            loss = criterion(selected_true_values, selected_predictions)
            losses[loss_type].append(loss)
    return bin_content, bounds, losses
