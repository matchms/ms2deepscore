from typing import List

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.filtering.metadata_processing.add_fingerprint import _derive_fingerprint_from_smiles
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from tqdm import tqdm

from ms2deepscore.train_new_model.inchikey_pair_selection import select_inchi_for_unique_inchikeys
from ms2deepscore.vector_operations import cosine_similarity_matrix


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