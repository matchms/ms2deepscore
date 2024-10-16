import os
from typing import List
import numpy as np
import pandas as pd
import torch
from matchms.filtering.metadata_processing.add_fingerprint import \
    _derive_fingerprint_from_smiles
from matchms.similarity.vector_similarity_functions import \
    jaccard_similarity_matrix
from matchms.Spectrum import Spectrum
from tqdm import tqdm
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from ms2deepscore.train_new_model.inchikey_pair_selection import \
    select_inchi_for_unique_inchikeys
from ms2deepscore.utils import save_pickled_file


def calculate_true_values_and_predictions_for_validation_spectra(positive_validation_spectra: List[Spectrum],
                                                                 negative_validation_spectra: List[Spectrum],
                                                                 ms2deepsore_model_file_name,
                                                                 computed_scores_directory = None,
                                                                 ):
    validation_spectra = {"positive": positive_validation_spectra,
                          "negative": negative_validation_spectra,
                          "both": positive_validation_spectra + negative_validation_spectra}
    # Load in MS2Deepscore model
    ms2deepscore_model = MS2DeepScore(load_model(ms2deepsore_model_file_name))

    possible_comparisons = (("positive", "positive"),
                            ("negative", "positive"),
                            ("negative", "negative"),
                            ("both", "both"))

    true_values_collection = {}
    predictions_collection = {}
    for ionmode_1, ionmode_2 in possible_comparisons:
        if computed_scores_directory is not None:
            os.makedirs(computed_scores_directory, exist_ok=True)
            file_name_true_values = os.path.join(computed_scores_directory, f"{ionmode_1}_{ionmode_2}_true_values.pickle")
            file_name_predictions = os.path.join(computed_scores_directory, f"{ionmode_1}_{ionmode_2}_predictions.pickle")
            if os.path.exists(file_name_true_values) or os.path.exists(file_name_predictions):
                raise FileExistsError

        true_values = get_tanimoto_score_between_spectra(validation_spectra[ionmode_1],
                                                         validation_spectra[ionmode_2])
        true_values_collection[f"{ionmode_1}_{ionmode_2}"] = true_values
        if computed_scores_directory is not None:
            save_pickled_file(true_values, file_name_true_values)

        predictions = ms2deepscore_model.matrix(
            validation_spectra[ionmode_1],
            validation_spectra[ionmode_2],
            is_symmetric=(validation_spectra[ionmode_1] == validation_spectra[ionmode_2]))
        predictions_collection[f"{ionmode_1}_{ionmode_2}"] = predictions
        if computed_scores_directory is not None:
            save_pickled_file(predictions, file_name_predictions)

    # Avoid memory leakage
    torch.cuda.empty_cache()
    del ms2deepscore_model

    return true_values_collection, predictions_collection


def calculate_tanimoto_scores_unique_inchikey(list_of_spectra_1: List[Spectrum],
                                              list_of_spectra_2: List[Spectrum],
                                              fingerprint_type="daylight",
                                              nbits=2048
                                              ):
    """Returns a dataframe with the tanimoto scores between each unique inchikey in list of spectra"""

    def get_fingerprint(smiles: str):
        fingerprint = _derive_fingerprint_from_smiles(smiles,
                                                      fingerprint_type=fingerprint_type,
                                                      nbits=nbits)
        assert isinstance(fingerprint, np.ndarray), f"Fingerprint could not be set smiles is {smiles}"
        return fingerprint

    if len(list_of_spectra_1) == 0 or len(list_of_spectra_2) == 0:
        raise ValueError("The nr of spectra to calculate tanimoto scores should be larger than 0")

    spectra_with_most_frequent_inchi_per_inchikey_1, unique_inchikeys_1 = \
        select_inchi_for_unique_inchikeys(list_of_spectra_1)
    spectra_with_most_frequent_inchi_per_inchikey_2, unique_inchikeys_2 = \
        select_inchi_for_unique_inchikeys(list_of_spectra_2)

    list_of_smiles_1 = [spectrum.get("smiles") for spectrum in spectra_with_most_frequent_inchi_per_inchikey_1]
    list_of_smiles_2 = [spectrum.get("smiles") for spectrum in spectra_with_most_frequent_inchi_per_inchikey_2]

    fingerprints_1 = np.array([get_fingerprint(spectrum) for spectrum in tqdm(list_of_smiles_1,
                                                                              desc="Calculating fingerprints")])
    fingerprints_2 = np.array([get_fingerprint(spectrum) for spectrum in tqdm(list_of_smiles_2,
                                                                              desc="Calculating fingerprints")])
    print("Calculating tanimoto scores")
    tanimoto_scores = jaccard_similarity_matrix(fingerprints_1, fingerprints_2)
    tanimoto_df = pd.DataFrame(tanimoto_scores, index=unique_inchikeys_1, columns=unique_inchikeys_2)
    return tanimoto_df


def get_tanimoto_score_between_spectra(spectra_1: List[Spectrum],
                                       spectra_2: List[Spectrum],
                                       fingerprint_type="daylight",
                                       nbits=2048):
    """Gets the tanimoto scores between two list of spectra

    It is optimized by calculating the tanimoto scores only between unique fingerprints/smiles.
    The tanimoto scores are derived after.

    """
    tanimoto_df = calculate_tanimoto_scores_unique_inchikey(spectra_1, spectra_2,
                                                            fingerprint_type,
                                                            nbits)
    inchikeys_1 = [spectrum.get("inchikey")[:14] for spectrum in spectra_1]
    inchikeys_2 = [spectrum.get("inchikey")[:14] for spectrum in spectra_2]
    tanimoto_scores = tanimoto_df.loc[inchikeys_1, inchikeys_2].values
    return tanimoto_scores
