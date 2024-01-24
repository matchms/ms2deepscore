import numpy as np
import torch
from matchms.similarity.vector_similarity_functions import cosine_similarity_matrix
from torch import nn

from ms2deepscore.benchmarking.calculate_scores_for_validation import get_tanimoto_score_between_spectra
from ms2deepscore.models import SiameseSpectralModel
from ms2deepscore.models.SiameseSpectralModel import compute_embedding_array


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
        Bins for the refernce score to evaluate the performance of scores.
    loss_types
        Specify list of loss types out of "mse", "mae", "rmse".
    """
    bin_content = []
    losses = {"bin": []}
    for loss_type in loss_types:
        if loss_type not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss function: {loss_type}. Must be one of: {LOSS_FUNCTIONS.keys()}")
        losses[loss_type] = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    for i in range(len(ref_scores_bins_inclusive) - 1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i + 1]
        bounds.append((low, high))
        idx = np.where((true_values >= low) & (true_values < high))
        bin_content.append(idx[0].shape[0])
        # Add values
        losses["bin"].append((low, high))
        for loss_type in loss_types:
            criterion = LOSS_FUNCTIONS[loss_type]
            loss = criterion(true_values[idx], predictions[idx])
            losses[loss_type].append(loss)
    return bin_content, bounds, losses


class ValidationLossCalculator:
    def __init__(self,
                 val_spectrums,
                 score_bins):
        # Check if the spectra only are unique inchikeys
        inchikeys_list = [s.get("inchikey") for s in val_spectrums]
        assert len(set(inchikeys_list)) == len(val_spectrums), 'Expected 1 spectrum per inchikey'
        self.target_scores = get_tanimoto_score_between_spectra(val_spectrums, val_spectrums)
        self.val_spectrums = val_spectrums
        self.score_bins = score_bins

    def compute_binned_validation_loss(self,
                                       model: SiameseSpectralModel,
                                       loss_types):
        """Benchmark the model against a validation set.
        """
        embeddings = compute_embedding_array(model, self.val_spectrums)
        ms2ds_scores = cosine_similarity_matrix(embeddings, embeddings)
        losses = bin_dependent_losses(ms2ds_scores,
                                      self.target_scores,
                                      self.score_bins,
                                      loss_types=loss_types
                                      )
        average_losses = {}
        for loss_type in losses:
            average_losses[loss_type] = np.mean(losses[loss_type])
        return average_losses