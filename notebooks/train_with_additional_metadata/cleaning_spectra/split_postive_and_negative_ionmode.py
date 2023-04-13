from tqdm import tqdm
import os
from typing import List, Union
import pickle
from matchms import importing
from matchms import Spectrum


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def load_spectra(file_name
                 ) -> Union[List[Spectrum], None]:
    """Will already be implemented in newer versions of matchms.
    Loads spectra from your spectrum file into memory as matchms Spectrum object

    The following file extensions can be loaded in with this function:
    "mzML", "json", "mgf", "msp", "mzxml", "usi" and "pickle".
    A pickled file is expected to directly contain a list of matchms spectrum objects.

    Args:
    -----
    file_name:
        Path to file containing spectra, with file extension "mzML", "json", "mgf", "msp",
        "mzxml", "usi" or "pickle"
    """
    assert os.path.exists(file_name), f"The specified file: {file_name} does not exists"

    file_extension = os.path.splitext(file_name)[1].lower()
    if file_extension == ".mzml":
        return list(importing.load_from_mzml(file_name))
    if file_extension == ".json":
        return list(importing.load_from_json(file_name))
    if file_extension == ".mgf":
        return list(importing.load_from_mgf(file_name))
    if file_extension == ".msp":
        return list(importing.load_from_msp(file_name))
    if file_extension == ".mzxml":
        return list(importing.load_from_mzxml(file_name))
    if file_extension == ".usi":
        return list(importing.load_from_usi(file_name))
    if file_extension == ".pickle":
        spectra = load_pickled_file(file_name)
        assert isinstance(spectra, list), "Expected list of spectra"
        assert isinstance(spectra[0], Spectrum), "Expected list of spectra"
        return spectra
    assert False, f"File extension of file: {file_name} is not recognized"


def remove_wrong_ion_modes(spectra: List[Spectrum],
                           ion_mode_to_keep):
    """Removes spectra that are not in the correct ionmode"""
    assert ion_mode_to_keep in {"positive", "negative"}, "ion_mode should be set to 'positive' or 'negative'"
    spectra_to_keep = []
    for spectrum in tqdm(spectra,
                         desc=f"Selecting {ion_mode_to_keep} mode spectra"):
        if spectrum is not None:
            if spectrum.get("ionmode") == ion_mode_to_keep:
                spectra_to_keep.append(spectrum)
    print(f"From {len(spectra)} spectra, "
          f"{len(spectra) - len(spectra_to_keep)} are removed since they are not in {ion_mode_to_keep} mode")
    return spectra_to_keep


def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    spectra = load_spectra("../../../../data/all_cleaned_spectra.pickle")
    print(len(spectra))
    positive_mode = remove_wrong_ion_modes(spectra, "positive")
    save_pickled_file(positive_mode, "../../../../data/positive_mode_spectra.pickle")
    print(len(positive_mode))
    negative_mode = remove_wrong_ion_modes(spectra, "negative")
    save_pickled_file(negative_mode, "../../../../data/negative_mode_spectra.pickle")
    print(len(negative_mode))
