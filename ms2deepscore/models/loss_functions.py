import torch


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


LOSS_FUNCTIONS = {
    "mse": mse_loss,
    "mae": mae_loss,
    "rmse": rmse_loss,
    "risk_mae": risk_aware_mae,
    "risk_mse": risk_aware_mse,
}
