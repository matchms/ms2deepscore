"""Custom loss and helper function."""
import torch


def l1_regularization(model, lambda_l1):
    """L1 regulatization for first dense layer of model."""
    l1_loss = torch.linalg.vector_norm(next(model.encoder.dense_layers[0].parameters()), ord=1)
    return lambda_l1 * l1_loss


def l2_regularization(model, lambda_l2):
    """L2 regulatization for first dense layer of model."""
    l2_loss = torch.linalg.vector_norm(next(model.encoder.dense_layers[0].parameters()), ord=2)
    return lambda_l2 * l2_loss


def risk_aware_loss(outputs, targets, percentiles=None):
    """Higher linear loss for predictions towards the majority of datapoints.

    Greedy implementation, using either actual percentiles or assuming a uniform
    distribution of datapoints within the range 0 to 1.
    """
    if percentiles is None:
        percentiles = torch.linspace(0.01, 1.0, 100)
    idx = []
    for target in targets:
        idx.append(torch.argmin(torch.abs(percentiles - target)))
    idx = torch.tensor(idx)
    max_bin = percentiles.shape[0]
    factors = (idx + 1) / max_bin

    errors = targets - outputs
    uppers =  factors * errors
    lowers = (factors - 1) * errors

    losses = torch.max(lowers, uppers)
    return losses.mean()


def mse_away_from_mean(output, target):
    """MSE weighted to get higher loss for predictions towards the mean of 0.5.
    
    In addition, we are usually more intereted in the precision for higher scores.
    And, we have often fewer pairs in that regime. This is included by an additional
    linear factor to shift attention to higher scores.
    """
    weighting = torch.exp(-10 * (output - 0.5)**2) + 1
    focus_high_scores = 1 + 0.5 * target
    loss = torch.mean(weighting * focus_high_scores * (output - target)**2)
    return loss
