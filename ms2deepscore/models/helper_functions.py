"""Custom loss and helper function."""
import torch


def initialize_device():
    """Initialize and return the device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will happen on {device}.")
    return device


def l1_regularization(model, lambda_l1):
    """L1 regulatization for first dense layer of model."""
    l1_loss = torch.linalg.vector_norm(next(model.encoder.dense_layers[0].parameters()), ord=1)
    return lambda_l1 * l1_loss


def l2_regularization(model, lambda_l2):
    """L2 regulatization for first dense layer of model."""
    l2_loss = torch.linalg.vector_norm(next(model.encoder.dense_layers[0].parameters()), ord=2)
    return lambda_l2 * l2_loss
