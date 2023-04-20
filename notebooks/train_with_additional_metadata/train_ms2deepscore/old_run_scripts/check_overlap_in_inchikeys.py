import numpy as np
import pickle
from tqdm import tqdm

def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


binned_spectra = load_pickled_file("../../../data/all_cleaned_spectra.pickle")
tanimoto_scores = load_pickled_file("../../../data/all_spectra_230201_annotated_min2peaks_tanimoto_scores.pickle")
print("loaded stuff")
inchikeys_df = set(tanimoto_scores.index)
spectrum_inchikeys = set([s.get("inchikey")[:14] for s in tqdm(binned_spectra)])
missing = (inchikeys_df ^ spectrum_inchikeys) & spectrum_inchikeys
print(missing)
print(len(missing))
print(len(inchikeys_df))
print(len(spectrum_inchikeys))