import random
from typing import List, Tuple
from matchms import Spectrum
from tqdm import tqdm


def select_unique_inchikeys(spectra):
    """Creates a list with all the unique first 14 character inchikeys"""
    inchikey_list = []
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")[:14]
        inchikey_list.append(inchikey)
    inchikey_set = set(inchikey_list)
    return sorted(list(inchikey_set))


def select_spectra_belonging_to_inchikey(
    spectra: List[Spectrum], inchikeys: List[str]
) -> List[Spectrum,]:
    # Select spectra belonging to the selected inchikeys
    spectra_containing_inchikey = []
    for spectrum in tqdm(spectra, desc="Finding spectra belonging to inchikeys"):
        inchikey = spectrum.get("inchikey")[:14]
        if inchikey in inchikeys:
            spectra_containing_inchikey.append(spectrum)
    return spectra_containing_inchikey


def split_spectra_in_random_inchikey_sets(
    spectra: List[Spectrum], k: int, random_seed
) -> Tuple[List[Spectrum], List[Spectrum], List[Spectrum]]:
    """Splits a set of spectra into a val, test and train set. The size of the val and test set are n/k.
    """
    unique_inchikeys = select_unique_inchikeys(spectra)
    random.seed(random_seed)
    random.shuffle(unique_inchikeys)
    fraction_size = len(unique_inchikeys) // k

    validation_inchikeys = unique_inchikeys[-fraction_size:]
    test_inchikeys = unique_inchikeys[:fraction_size]
    train_inchikeys = unique_inchikeys[fraction_size:-fraction_size]
    if len(unique_inchikeys) != (len(validation_inchikeys) + len(test_inchikeys) + len(train_inchikeys)):
        raise ValueError("Unexpected number of unique inchikeys.")

    validation_spectra = select_spectra_belonging_to_inchikey(
        spectra, validation_inchikeys
    )
    test_spectra = select_spectra_belonging_to_inchikey(spectra, test_inchikeys)
    train_spectra = select_spectra_belonging_to_inchikey(spectra, train_inchikeys)
    if len(spectra) != (len(validation_spectra) + len(test_spectra) + len(train_spectra)):
        raise ValueError("Unexpected number of unique inchikeys.")

    return validation_spectra, test_spectra, train_spectra
