import pickle
import os
import matchms.filtering as msfilters
from tqdm import tqdm

def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


data_folder = "../../data"
file_name = "all_spectra_230201_matchms_manual_pubchem.pickle"
library_spectra = load_pickled_file(os.path.join(data_folder, file_name))

def minimal_number_of_peaks(s):
    s = msfilters.select_by_mz(s, mz_from=0.0, mz_to=1000.0)
    s = msfilters.require_minimum_number_of_peaks(s, n_required=3)
    return s

clean_spectra = [minimal_number_of_peaks(s) for s in tqdm(library_spectra)]
clean_spectra_not_none = [s for s in clean_spectra if s is not None]


def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


file_name = os.path.join(data_folder, "all_cleaned_spectra.pickle")

save_pickled_file(clean_spectra_not_none, file_name)
