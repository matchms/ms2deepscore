import torch

from ms2deepscore import MS2DeepScore
from ms2deepscore.validation_loss_calculation.PredictionsAndTanimotoScores import PredictionsAndTanimotoScores
from ms2deepscore.validation_loss_calculation.calculate_scores_for_validation import create_embedding_matrix_symmetric, \
    create_embedding_matrix_not_symmetric, calculate_tanimoto_scores_unique_inchikey
from ms2deepscore.models.load_model import load_model


class CalculateScoresBetweenAllIonmodes:
    """Calculates the true tanimoto scores and scores between all ionmodes"""
    def __init__(self,
                 model_file_name, positive_validation_spectra, negative_validation_spectra,
                 fingerprint_type="daylight", n_bits_fingerprint=2048):
        self.model_file_name = model_file_name
        self.postive_validation_spectra = positive_validation_spectra
        self.negative_validation_spectra = negative_validation_spectra
        self.model = MS2DeepScore(load_model(model_file_name))

        self.pos_vs_neg_scores = self.get_tanimoto_and_prediction_pairs(
            positive_validation_spectra, negative_validation_spectra, label="positive vs negative",
            fingerprint_type=fingerprint_type, n_bits=n_bits_fingerprint)
        self.pos_vs_pos_scores = self.get_tanimoto_and_prediction_pairs(
            positive_validation_spectra, label="positive vs positive",
            fingerprint_type=fingerprint_type, n_bits=n_bits_fingerprint)
        self.neg_vs_neg_scores = self.get_tanimoto_and_prediction_pairs(
            negative_validation_spectra, label="negative vs negative",
            fingerprint_type=fingerprint_type, n_bits=n_bits_fingerprint)
        # Avoid memory leakage
        torch.cuda.empty_cache()
        del self.model

    def get_tanimoto_and_prediction_pairs(self, spectra_1, spectra_2=None, label="",
                                          fingerprint_type="daylight",
                                          n_bits=2048) -> PredictionsAndTanimotoScores:
        symmetric = False
        if spectra_2 is None:
            spectra_2 = spectra_1
            symmetric = True
        if symmetric:
            predictions_df = create_embedding_matrix_symmetric(self.model, spectra_1)
        else:
            predictions_df = create_embedding_matrix_not_symmetric(self.model, spectra_1, spectra_2)
        tanimoto_scores_df = calculate_tanimoto_scores_unique_inchikey(spectra_1, spectra_2,
                                                                       fingerprint_type=fingerprint_type, nbits=n_bits)
        return PredictionsAndTanimotoScores(predictions_df, tanimoto_scores_df, symmetric, label)

    def list_of_predictions_and_tanimoto_scores(self):
        return [self.pos_vs_pos_scores,
                self.pos_vs_neg_scores,
                self.neg_vs_neg_scores, ]
