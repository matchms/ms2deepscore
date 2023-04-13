from ms2deepscore.train_new_model.calculate_tanimoto_matrix import calculate_tanimoto_scores_unique_inchikey

if __name__ == "__main__":
    import pickle
    import os
    def load_pickled_file(filename: str):
        with open(filename, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object
    all_spectra = load_pickled_file("../../../../data/cleaned_spectra/negative_annotated_spectra.pickle")
    print("loaded in spectra")
    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(all_spectra, all_spectra)

    def save_pickled_file(obj, filename: str):
        assert not os.path.exists(filename), "File already exists"
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    save_pickled_file(tanimoto_scores, "../../../../data/tanimoto_scores/negative_tanimoto_scores.pickle")