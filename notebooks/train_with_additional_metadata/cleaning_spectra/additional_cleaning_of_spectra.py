import pickle
import os
from typing import List, Tuple
import matchms.filtering as msfilters
from tqdm import tqdm
from matchms.metadata_utils import is_valid_inchi, is_valid_inchikey, is_valid_smiles
from matchms import Spectrum


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


data_folder = "../../../data"
file_name = "all_spectra_230201_matchms_manual_pubchem.pickle"
library_spectra = load_pickled_file(os.path.join(data_folder, file_name))


def minimal_number_of_peaks(s):
    s = msfilters.select_by_mz(s, mz_from=0.0, mz_to=1000.0)
    s = msfilters.require_minimum_number_of_peaks(s, n_required=3)
    return s


def check_fully_annotated(spectrum: Spectrum) -> bool:
    if not is_valid_smiles(spectrum.get("smiles")):
        return False
    if not is_valid_inchikey(spectrum.get("inchikey")):
        return False
    if not is_valid_inchi(spectrum.get("inchi")):
        return False
    spectrum = msfilters.require_precursor_mz(spectrum)
    if spectrum is None:
        return False
    return True


def split_annotated_spectra(spectra: List[Spectrum]) -> Tuple[List[Spectrum], List[Spectrum]]:
    fully_annotated_spectra = []
    not_fully_annotated_spectra = []
    for spectrum in tqdm(spectra,
                         desc="Splitting annotated and unannotated spectra"):
        fully_annotated = check_fully_annotated(spectrum)
        if fully_annotated:
            fully_annotated_spectra.append(spectrum)
        else:
            not_fully_annotated_spectra.append(spectrum)
    print(f"From {len(spectra)} spectra, "
          f"{len(spectra) - len(fully_annotated_spectra)} are removed since they are not fully annotated")
    return fully_annotated_spectra, not_fully_annotated_spectra


clean_spectra = [s for s in tqdm(library_spectra) if s.get("ionmode") in ["positive", "negative"]]
clean_spectra = [minimal_number_of_peaks(s) for s in tqdm(clean_spectra)]
clean_spectra_not_none = [s for s in clean_spectra if s is not None]

fully_annotated_spectra, not_fully_annotated_spectra = split_annotated_spectra(clean_spectra_not_none)

def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


file_name = os.path.join(data_folder, "all_cleaned_spectra.pickle")

save_pickled_file(fully_annotated_spectra, file_name)
