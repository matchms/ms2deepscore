from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.utils import create_evenly_spaced_bins


def plot_average_per_bin(scores_between_ionmodes: CalculateScoresBetweenAllIonmodes, nr_of_bins):
    bins = create_evenly_spaced_bins(nr_of_bins)
    bin_centers = [(bin_borders[0] + bin_borders[1])/2 for bin_borders in bins]
    fig, ax = plt.subplots()

    for predictions_and_tanimoto_scores in scores_between_ionmodes.list_of_predictions_and_tanimoto_scores():
        average_predictions = predictions_and_tanimoto_scores.get_average_prediction_per_inchikey_pair()
        _, average_per_bin = predictions_and_tanimoto_scores.get_average_per_bin(average_predictions, bins)
        ax.plot(bin_centers, average_per_bin, label=predictions_and_tanimoto_scores.label)

    ax.set_xlabel("True chemical similarity")
    ax.set_ylabel("Average predicted chemical similarity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return fig
