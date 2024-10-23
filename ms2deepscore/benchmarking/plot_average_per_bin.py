import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.benchmarking.PredictionsAndTanimotoScores import PredictionsAndTanimotoScores


def select_pairs_per_bin(predictions_and_tanimoto_scores: PredictionsAndTanimotoScores, bins):
    digitized = np.digitize(predictions_and_tanimoto_scores.list_of_tanimoto_scores, bins) - 1
    average_per_bin = []
    for bin in tqdm(range(len(bins)-1), desc="Selecting available inchikey pairs per bin"):
        predictions_in_this_bin = []
        indexes_of_pairs_in_bin = np.where(digitized == bin)[0]
        for i in indexes_of_pairs_in_bin:
            predictions_in_this_bin.append(predictions_and_tanimoto_scores.list_of_average_predictions[i])
        if len(predictions_in_this_bin) == 0:
            raise ValueError(f"The bin between {bins[bin]} - {bins[bin + 1]}does not have any pairs")
        average_per_bin.append(sum(predictions_in_this_bin)/len(predictions_in_this_bin))
    return average_per_bin


def plot_average_per_bin(pairs: CalculateScoresBetweenAllIonmodes, nr_of_bins):
    bins = np.linspace(0, 1.00000001, nr_of_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers for plotting

    pos_pos_average_per_bin = select_pairs_per_bin(pairs.pos_vs_pos_scores, bins)
    pos_neg_average_per_bin = select_pairs_per_bin(pairs.pos_vs_neg_scores, bins)
    neg_neg_average_per_bin = select_pairs_per_bin(pairs.neg_vs_neg_scores, bins)
    fig, ax = plt.subplots()
    ax.plot(bin_centers, neg_neg_average_per_bin, label="Negative vs negative")
    ax.plot(bin_centers, pos_pos_average_per_bin, label="Positive vs positive")
    ax.plot(bin_centers, pos_neg_average_per_bin, label="Positive vs negative")

    ax.set_xlabel("True chemical similarity")
    ax.set_ylabel("Average predicted chemical similarity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return fig
