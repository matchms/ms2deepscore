import pandas as pd
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from ms2deepscore.train_new_model.inchikey_pair_selection import select_inchi_for_unique_inchikeys, \
    compute_fingerprints_for_training
from matchms.importing.load_from_mgf import load_from_mgf
from tqdm import tqdm

def calculate_tanimoto_scores_unique_inchikey(list_of_spectra,
    fingerprint_type="daylight",
    nbits=2048
    ) -> pd.DataFrame:
    fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
        list_of_spectra,
        fingerprint_type,
        nbits
        )
    tanimoto_scores = jaccard_similarity_matrix(fingerprints, fingerprints)
    tanimoto_df = pd.DataFrame(tanimoto_scores, index=inchikeys14_unique, columns=inchikeys14_unique)
    return tanimoto_df


neg_training_spectra = list(tqdm(load_from_mgf("/lustre/BIF/nobackup/jonge094/ms2deepscore/data/pytorch/new_corinna_included/training_and_validation_split/negative_validation_spectra.mgf")))

tanimoto_df_training_spectra = calculate_tanimoto_scores_unique_inchikey(
    neg_training_spectra,
    fingerprint_type="daylight",
    nbits=4096)

tanimoto_df_training_spectra.to_csv("tanimoto_scores_validation_spectra_neg.csv")

pos_training_spectra = list(load_from_mgf("/lustre/BIF/nobackup/jonge094/ms2deepscore/data/pytorch/new_corinna_included/training_and_validation_split/positive_validation_spectra.mgf"))

tanimoto_df_training_spectra = calculate_tanimoto_scores_unique_inchikey(
    pos_training_spectra,
    fingerprint_type="daylight",
    nbits=4096)

tanimoto_df_training_spectra.to_csv("tanimoto_scores_validation_spectra_pos.csv")
