from collections import Counter
from typing import List, Tuple
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.filtering import add_fingerprint
from matchms.similarity.vector_similarity_functions import \
    jaccard_similarity_matrix
from rdkit import Chem
from tqdm import tqdm


def compute_fingerprints_for_training(spectrums,
                                      fingerprint_type: str = "daylight",
                                      nbits: int = 2048):
    """Calculates fingerprints for each unique inchikey and removes spectra for which no fingerprint could be created"""
    spectra_selected, inchikeys14_unique = select_inchi_for_unique_inchikeys(spectrums)
    print(f"Selected {len(spectra_selected)} spectra with unique inchikeys (out of {len(spectrums)} spectra)")
    # Compute fingerprints using matchms
    spectra_selected = [add_fingerprint(s, fingerprint_type, nbits)\
                        if s.get("fingerprint") is None else s for s in spectra_selected]

    # Ignore missing / not-computed fingerprints
    fingerprints = [s.get("fingerprint") for s in tqdm(spectra_selected,
                                                       desc="Calculating fingerprints")]
    idx = np.array([i for i, x in enumerate(fingerprints) if x is not None]).astype(int)
    if len(idx) == 0:
        raise ValueError("No fingerprints could be computed")
    if len(idx) < len(fingerprints):
        print(f"Successfully generated fingerprints for {len(idx)} of {len(fingerprints)} spectra")
    fingerprints = np.array([fingerprints[i] for i in idx])
    inchikeys14_unique = [inchikeys14_unique[i] for i in idx]
    spectra_selected = [spectra_selected[i] for i in idx]
    return fingerprints, inchikeys14_unique, spectra_selected


def select_inchi_for_unique_inchikeys(
    list_of_spectra: List['Spectrum']
) -> Tuple[List['Spectrum'], List[str]]:
    """Select spectra with most frequent inchi for unique inchikeys.

    Method needed to calculate Tanimoto scores.
    """
    # Extract inchi's and inchikeys from spectra metadata
    inchikeys_list = [s.get("inchikey") for s in list_of_spectra]
    inchi_list = [s.get("inchi") for s in list_of_spectra]

    inchi_array = np.array(inchi_list)
    inchikeys14_array = np.array([x[:14] for x in inchikeys_list])

    # Find unique inchikeys
    inchikeys14_unique = sorted(set(inchikeys14_array))

    spectra_selected = []
    for inchikey14 in inchikeys14_unique:
        # Indices of matching inchikeys
        idx = np.where(inchikeys14_array == inchikey14)[0]

        # Find the most frequent inchi for the inchikey
        most_common_inchi = Counter(inchi_array[idx]).most_common(1)[0][0]

        # ID of the spectrum with the most frequent inchi
        ID = idx[np.where(inchi_array[idx] == most_common_inchi)[0][0]]

        spectra_selected.append(list_of_spectra[ID].clone())

    return spectra_selected, inchikeys14_unique


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


def get_tanimoto_score_between_spectra(tanimoto_df, reference_spectra_1, reference_spectra_2):
    """Retrieves the tanimoto score from a tanimoto score df."""
    def get_tanimoto_indexes(tanimoto_df, spectra):
        inchikey_idx_reference_spectra_1 = np.zeros(len(spectra))
        for i, spec in enumerate(spectra):
            inchikey_idx_reference_spectra_1[i] = np.where(tanimoto_df.index.values == spec.get("inchikey")[:14])[0]
        return inchikey_idx_reference_spectra_1.astype("int")
    inchikey_idx_1 = get_tanimoto_indexes(tanimoto_df, reference_spectra_1)
    inchikey_idx_2 = get_tanimoto_indexes(tanimoto_df, reference_spectra_2)

    scores_ref = tanimoto_df.values[np.ix_(inchikey_idx_1[:], inchikey_idx_2[:])].copy()
    return scores_ref