import pandas as pd
import numpy as np

from ms2deepscore import MS2DeepScore
from ms2deepscore.benchmarking.calculate_scores_for_validation import calculate_tanimoto_scores_unique_inchikey
from ms2deepscore.vector_operations import cosine_similarity_matrix
from ms2deepscore.models.load_model import load_model


class AveragePredictionAndTanimotoForInchikeyPairs:
    """Calculates the true tanimoto scores and average ms2deepscore between unique inchikeys """
    def __init__(self,
                 model_file_name, positive_validation_spectra, negative_validation_spectra):
        self.model_file_name = model_file_name
        self.postive_validation_spectra = positive_validation_spectra
        self.negative_validation_spectra = negative_validation_spectra
        self.model = MS2DeepScore(load_model(model_file_name))

        self.pos_vs_neg_predictions, self.pos_vs_neg_tanimoto_scores = self.get_tanimoto_and_prediction_pairs(
            positive_validation_spectra, negative_validation_spectra)
        self.pos_vs_pos_predictions, self.pos_vs_pos_tanimoto_scores = self.get_tanimoto_and_prediction_pairs(
            positive_validation_spectra)
        self.neg_vs_neg_predictions, self.neg_vs_neg_tanimoto_scores = self.get_tanimoto_and_prediction_pairs(
            negative_validation_spectra)

    def get_tanimoto_and_prediction_pairs(self, spectra_1, spectra_2 = None):
        symmetric = False
        if spectra_2 is None:
            spectra_2 = spectra_1
            symmetric = True
        if symmetric:
            predictions_df = create_embedding_matrix_symmetric(self.model, spectra_1)
        else:
            predictions_df = create_embedding_matrix_not_symmetric(self.model, spectra_1, spectra_2)
        predictions_per_inchikey = convert_predictions_matrix_to_average_per_inchikey(predictions_df,
                                                                                      symmetric=symmetric)
        tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(spectra_1, spectra_2)
        predictions, tanimoto_scores = select_score_pairs(predictions_per_inchikey, tanimoto_scores)
        return predictions, tanimoto_scores


def select_score_pairs(predictions_per_inchikey, tanimoto_df):
    predictions = []
    tanimoto_scores = []
    for inchikey_1 in predictions_per_inchikey.index:
        for inchikey_2 in  predictions_per_inchikey.index:
            prediction = predictions_per_inchikey[inchikey_2][inchikey_1]
            # don't include pairs where the prediciton is Nan (this is the case when only a pair against itself is available)
            if not np.isnan(prediction):
                tanimoto = tanimoto_df[inchikey_2][inchikey_1]
                predictions.append(prediction)
                tanimoto_scores.append(tanimoto)
    return predictions, tanimoto_scores


# Functions for creating predictions and true value matrix
def create_embedding_matrix_symmetric(model, spectra):
    print("Calculating embeddings")
    embeddings = model.get_embedding_array(spectra)
    print("Calculating similarity between embeddings")
    predictions = cosine_similarity_matrix(embeddings, embeddings)
    # Select the inchikeys per spectrum
    inchikeys = []
    for spectrum in spectra:
        inchikeys.append(spectrum.get("inchikey")[:14])
    # create dataframe with inchikeys as indexes
    predictions_df = pd.DataFrame(predictions, index=inchikeys, columns=inchikeys)
    return predictions_df


def create_embedding_matrix_not_symmetric(model, spectra, spectra_2):
    print("Calculating embeddings")
    embeddings1 = model.get_embedding_array(spectra)
    embeddings2 = model.get_embedding_array(spectra_2)
    print("Calculating similarity between embeddings")

    predictions = cosine_similarity_matrix(embeddings1, embeddings2)
    # Select the inchikeys per spectrum
    inchikeys1 = [spectrum.get("inchikey")[:14] for spectrum in spectra]
    inchikeys2 = [spectrum.get("inchikey")[:14] for spectrum in spectra_2]

    # create dataframe with inchikeys as indexes
    predictions_df = pd.DataFrame(predictions, index=inchikeys1, columns=inchikeys2)
    return predictions_df


def convert_predictions_matrix_to_average_per_inchikey(predictions_df, symmetric=True):
    """Takes a matrix with per spectrum predictions and converts it to a df with the average prediction between all inchikeys"""
    # remove predicitons between the same spectrum
    if symmetric:
        np.fill_diagonal(predictions_df.values, np.nan)
    # get the mean prediction per inchikey
    df_grouped = predictions_df.groupby(
        predictions_df.index).mean()  # Grouping rows by index and averaging across columns
    df_grouped_columns = df_grouped.groupby(lambda x: x, axis=1).mean()  # Grouping columns with duplicate names
    return df_grouped_columns


# # functions for sampling pairs per bin equally

#
#
#
#
# class ReshuffleSampler:
#     def __init__(self, values):
#         self.original_values = values
#         self.current_values = []
#         self.shuffle()
#
#     def shuffle(self):
#         """Shuffle the list and reset the current values."""
#         self.current_values = self.original_values.copy()
#         random.shuffle(self.current_values)
#
#     def sample(self):
#         """Sample a value, reshuffle if list is exhausted."""
#         if not self.current_values:
#             self.shuffle()  # Reshuffle when the list is exhausted
#         return self.current_values.pop()  # Pop a value from the list
#
#
# def sample_pairs_per_bin_equally(score_pairs_per_bin, nr_of_pairs_per_bin):
#     selected_pairs = []
#     for pairs_in_bin in tqdm(score_pairs_per_bin, desc="Sample pairs per bin"):
#         sampler = ReshuffleSampler(pairs_in_bin)
#
#         for i in range(nr_of_pairs_per_bin):
#             selected_pairs.append(sampler.sample())
#     return selected_pairs
#
