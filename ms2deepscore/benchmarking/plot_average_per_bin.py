import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.validation_loss_calculation.PredictionsAndTanimotoScores import PredictionsAndTanimotoScores


def select_pairs_per_bin(predictions_and_tanimoto_scores: PredictionsAndTanimotoScores, bins):
    average_predictions = predictions_and_tanimoto_scores.get_average_prediction_per_inchikey_pair().to_numpy()

    digitized = np.digitize(predictions_and_tanimoto_scores.tanimoto_df, bins) - 1
    average_per_bin = []
    for bin in tqdm(range(len(bins)-1), desc="Selecting available inchikey pairs per bin"):
        row_idxs, col_idxs = np.where(digitized == bin)
        predictions_in_this_bin = average_predictions[row_idxs, col_idxs]
        if len(predictions_in_this_bin) == 0:
            average_per_bin.append(0)
            print(f"The bin between {bins[bin]} - {bins[bin + 1]}does not have any pairs")
        else:
            average_per_bin.append(predictions_in_this_bin.mean())
    return average_per_bin


def plot_average_per_bin(scores_between_ionmodes: CalculateScoresBetweenAllIonmodes, nr_of_bins):
    bins = np.linspace(0, 1.00000001, nr_of_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers for plotting
    fig, ax = plt.subplots()

    for predictions_and_tanimoto_scores in scores_between_ionmodes.list_of_predictions_and_tanimoto_scores():
        average_per_bin = select_pairs_per_bin(predictions_and_tanimoto_scores, bins)
        ax.plot(bin_centers, average_per_bin, label=predictions_and_tanimoto_scores.label)

    ax.set_xlabel("True chemical similarity")
    ax.set_ylabel("Average predicted chemical similarity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return fig
