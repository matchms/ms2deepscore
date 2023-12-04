from typing import List

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.similarity.vector_similarity_functions import \
    jaccard_similarity_matrix
from rdkit import Chem
from tqdm import tqdm

from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_inchi_for_unique_inchikeys


def calculate_tanimoto_scores_unique_inchikey(list_of_spectra_1: List[Spectrum],
                                              list_of_spectra_2: List[Spectrum]):
    """Returns a dataframe with the tanimoto scores between each unique inchikey in list of spectra"""

    def get_fingerprint(smiles: str):
        fingerprint = np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), fpSize=2048))
        assert isinstance(fingerprint, np.ndarray), f"Fingerprint could not be set smiles is {smiles}"
        return fingerprint

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
                                       spectra_2: List[Spectrum]):
    """Gets the tanimoto scores between two list of spectra

    It is optimized by calculating the tanimoto scores only between unique fingerprints/smiles.
    The tanimoto scores are derived after.

    """
    tanimoto_df = calculate_tanimoto_scores_unique_inchikey(spectra_1, spectra_2)
    inchikeys_1 = [spectrum.get("inchikey")[:14] for spectrum in spectra_1]
    inchikeys_2 = [spectrum.get("inchikey")[:14] for spectrum in spectra_2]
    tanimoto_scores = tanimoto_df.loc[inchikeys_1, inchikeys_2].values
    return tanimoto_scores