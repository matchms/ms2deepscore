import os
from ms2deepscore.utils import load_pickled_file, save_pickled_file, return_non_existing_file_name
from ms2deepscore.train_new_model.calculate_tanimoto_matrix import calculate_tanimoto_scores_unique_inchikey

if __name__ == "__main__":

    postive_spectra = load_pickled_file("../../../../data/training_and_validation_split/positive_validation_spectra.pickle")
    negative_spectra = load_pickled_file("../../../../data/training_and_validation_split/negative_validation_spectra.pickle")

    all_spectra = postive_spectra + negative_spectra

    print("loaded in spectra")
    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(all_spectra, all_spectra)

    save_pickled_file(tanimoto_scores, "../../../../data/tanimoto_scores/validation_tanimoto_scores.pickle")