import os
from typing import List
import torch
from matchms.Spectrum import Spectrum
from ms2deepscore import MS2DeepScore
from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import calculate_tanimoto_scores_unique_inchikey
from ms2deepscore.models import load_model
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
