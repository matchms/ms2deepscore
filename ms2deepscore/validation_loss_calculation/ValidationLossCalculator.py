import numpy as np

from ms2deepscore import MS2DeepScore
from ms2deepscore.validation_loss_calculation.calculate_scores_for_validation import create_embedding_matrix_symmetric, \
    calculate_tanimoto_scores_unique_inchikey
from ms2deepscore.validation_loss_calculation.PredictionsAndTanimotoScores import PredictionsAndTanimotoScores
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore


class ValidationLossCalculator:
    def __init__(self,
                 val_spectrums,
                 settings: SettingsMS2Deepscore):
        self.val_spectrums = val_spectrums
        self.score_bins = settings.same_prob_bins
        self.settings = settings
        self.tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(self.val_spectrums, self.val_spectrums,
                                                                         self.settings.fingerprint_type,
                                                                         self.settings.fingerprint_nbits)

    def compute_binned_validation_loss(self,
                                       model: SiameseSpectralModel,
                                       loss_types):
        """Benchmark the model against a validation set.
        """
        ms2deepscore_model = MS2DeepScore(model)
        ms2ds_scores = create_embedding_matrix_symmetric(ms2deepscore_model, self.val_spectrums)
        predictions_and_tanimoto_scores = PredictionsAndTanimotoScores(
            predictions_df=ms2ds_scores, tanimoto_df=self.tanimoto_scores, symmetric=True)
        losses_per_bin = {}
        for loss_type in loss_types:
            _, _, average_loss_per_bin = predictions_and_tanimoto_scores.get_average_loss_per_bin_per_inchikey_pair(
                loss_type,
                self.settings.same_prob_bins)
            losses_per_bin[loss_type] = average_loss_per_bin
        average_losses = {}
        for loss_type in loss_types:
            average_losses[loss_type] = np.mean(losses_per_bin[loss_type])
        return average_losses, losses_per_bin
