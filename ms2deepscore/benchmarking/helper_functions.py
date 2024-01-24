import numpy as np


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
    losses = {
        "bin": [],
        "mae": [],
        "mse": [],
        "rmse": [],
    }
    # maes = []
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
        if "mae" in loss_types:
            mae = np.abs(true_values[idx] - predictions[idx]).mean() 
            losses["mae"].append(mae)
        if "mse" in loss_types:
            mse = np.square(true_values[idx] - predictions[idx]).mean()
            losses["mse"].append(mse)
        if "rmse" in loss_types:
            rmse = np.sqrt(np.square(true_values[idx] - predictions[idx]).mean())
            losses["rmse"].append(rmse)
    return bin_content, bounds, losses
