import os
import pickle
from typing import Generator, List
from matchms import Spectrum
from matchms.importing import load_spectra


def create_peak_dict(peak_list):
    """ Create dictionary of merged peaks (keep max-intensity peak per bin).
    """
    peaks = {}
    for (ID, weight) in peak_list:
        if ID in peaks:
            peaks[ID] = max(weight, peaks[ID])
        else:
            peaks[ID] = weight
    return peaks


def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def return_non_existing_file_name(file_name):
    """Checks if a path already exists, otherwise creates a new filename with (1)"""
    if not os.path.exists(file_name):
        return file_name
    print(f"The file name already exists: {file_name}")
    file_name_base, file_extension = os.path.splitext(file_name)
    i = 1
    new_file_name = f"{file_name_base}({i}){file_extension}"
    while os.path.exists(new_file_name):
        i += 1
        new_file_name = f"{file_name_base}({i}){file_extension}"
    print(f"Instead the file will be stored in {new_file_name}")
    return new_file_name


def load_spectra_as_list(file_name) -> List[Spectrum]:
    spectra = load_spectra(file_name, metadata_harmonization=False)
    if isinstance(spectra, Generator):
        return list(spectra)
    return spectra
