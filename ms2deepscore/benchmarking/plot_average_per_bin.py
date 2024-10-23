import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.validation_loss_calculation.PredictionsAndTanimotoScores import PredictionsAndTanimotoScores


def select_pairs_per_bin(predictions_and_tanimoto_scores: PredictionsAndTanimotoScores, bins):
    digitized = np.digitize(predictions_and_tanimoto_scores.list_of_tanimoto_scores, bins) - 1
    average_per_bin = []
    for bin in tqdm(range(len(bins)-1), desc="Selecting available inchikey pairs per bin"):
        predictions_in_this_bin = []
        indexes_of_pairs_in_bin = np.where(digitized == bin)[0]
        for i in indexes_of_pairs_in_bin:
            predictions_in_this_bin.append(predictions_and_tanimoto_scores.list_of_average_predictions[i])
        if len(predictions_in_this_bin) == 0:
            average_per_bin.append(0)
            print(f"The bin between {bins[bin]} - {bins[bin + 1]}does not have any pairs")
        else:
            average_per_bin.append(sum(predictions_in_this_bin)/len(predictions_in_this_bin))
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
