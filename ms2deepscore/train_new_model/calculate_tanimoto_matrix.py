"""
This script is not needed for normally running MS2Query, it is only needed to generate a new library or to train
new models
"""
from collections import Counter
from typing import List

import pandas as pd
from matchms import Spectrum
import numpy as np

from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from rdkit import Chem
from tqdm import tqdm


def calculate_tanimoto_scores_unique_inchikey(list_of_spectra_1: List[Spectrum],
                                              list_of_spectra_2: List[Spectrum]):
    """Returns a dataframe with the tanimoto scores between each unique inchikey in list of spectra"""
    spectra_with_most_frequent_inchi_per_inchikey_1, unique_inchikeys_1 = select_inchi_for_unique_inchikeys(list_of_spectra_1)
    spectra_with_most_frequent_inchi_per_inchikey_2, unique_inchikeys_2 = select_inchi_for_unique_inchikeys(list_of_spectra_2)

    list_of_smiles_1 = [spectrum.get("smiles") for spectrum in spectra_with_most_frequent_inchi_per_inchikey_1]
    list_of_smiles_2 = [spectrum.get("smiles") for spectrum in spectra_with_most_frequent_inchi_per_inchikey_2]

    tanimoto_scores = calculate_tanimoto_scores_from_smiles(list_of_smiles_1, list_of_smiles_2)
    tanimoto_df = pd.DataFrame(tanimoto_scores, index=unique_inchikeys_1, columns=unique_inchikeys_2)
    return tanimoto_df


def select_inchi_for_unique_inchikeys(list_of_spectra: List[Spectrum]) -> (List[Spectrum], List[str]):
    """"Select spectra with most frequent inchi for unique inchikeys

    Method needed to calculate tanimoto scores"""
    # Select all inchi's and inchikeys from spectra metadata
    inchikeys_list = []
    inchi_list = []
    for s in list_of_spectra:
        inchikeys_list.append(s.get("inchikey"))
        inchi_list.append(s.get("inchi"))
    inchi_array = np.array(inchi_list)
    inchikeys14_array = np.array([x[:14] for x in inchikeys_list])

    # Select unique inchikeys
    inchikeys14_unique = sorted(list({x[:14] for x in inchikeys_list}))

    spectra_with_most_frequent_inchi_per_unique_inchikey = []
    for inchikey14 in inchikeys14_unique:
        # Select inchis for inchikey14
        idx = np.where(inchikeys14_array == inchikey14)[0]
        inchis_for_inchikey14 = [list_of_spectra[i].get("inchi") for i in idx]
        # Select the most frequent inchi per inchikey
        inchi = Counter(inchis_for_inchikey14).most_common(1)[0][0]
        # Store the ID of the spectrum with the most frequent inchi
        ID = idx[np.where(inchi_array[idx] == inchi)[0][0]]
        spectra_with_most_frequent_inchi_per_unique_inchikey.append(list_of_spectra[ID].clone())
    return spectra_with_most_frequent_inchi_per_unique_inchikey, inchikeys14_unique


def calculate_tanimoto_scores_from_smiles(list_of_smiles_1: List[str],
                                          list_of_smiles_2: List[str]) -> np.ndarray:
    """Returns a 2d ndarray containing the tanimoto scores between the smiles"""
    fingerprints_1 = np.array([get_fingerprint(spectrum) for spectrum in tqdm(list_of_smiles_1,
                                                                              desc="Calculating fingerprints")])
    fingerprints_2 = np.array([get_fingerprint(spectrum) for spectrum in tqdm(list_of_smiles_2,
                                                                              desc="Calculating fingerprints")])
    print("Calculating tanimoto scores")
    tanimoto_scores = jaccard_similarity_matrix(fingerprints_1, fingerprints_2)
    return tanimoto_scores


def get_fingerprint(smiles: str):
    fingerprint = np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), fpSize=2048))
    assert isinstance(fingerprint, np.ndarray), \
        f"Fingerprint for 1 spectrum could not be set smiles is {smiles}"
    return fingerprint
