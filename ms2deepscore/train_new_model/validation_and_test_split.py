import os
import random
from typing import List, Tuple
from matchms import Spectrum
from tqdm import tqdm
from ms2deepscore.utils import load_pickled_file, save_pickled_file


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
    spectra: List[Spectrum], k: int
) -> Tuple[List[Spectrum], List[Spectrum], List[Spectrum]]:
    """Splits a set of spectra into a val, test and train set. The size of the val and test set are n/k"""
    unique_inchikeys = select_unique_inchikeys(spectra)
    random.shuffle(unique_inchikeys)
    fraction_size = len(unique_inchikeys) // k

    validation_inchikeys = unique_inchikeys[-fraction_size:]
    test_inchikeys = unique_inchikeys[:fraction_size]
    train_inchikeys = unique_inchikeys[fraction_size:-fraction_size]
    assert len(unique_inchikeys) == len(
        validation_inchikeys + test_inchikeys + train_inchikeys
    )

    validation_spectra = select_spectra_belonging_to_inchikey(
        spectra, validation_inchikeys
    )
    test_spectra = select_spectra_belonging_to_inchikey(spectra, test_inchikeys)
    train_spectra = select_spectra_belonging_to_inchikey(spectra, train_inchikeys)
    assert len(spectra) == len(validation_spectra + test_spectra + train_spectra)

    return validation_spectra, test_spectra, train_spectra


def split_data_wrapper(spectra_location, data_directory, ionization_mode):
    spectra = list(load_pickled_file(spectra_location))
    random.seed(42)
    (
        validation_spectra,
        test_spectra,
        train_spectra,
    ) = split_spectra_in_random_inchikey_sets(spectra, 20)
    print(f"Nr of validation spectra: {len(validation_spectra)}")
    print(f"Nr of test spectra: {len(test_spectra)}")
    print(f"Nr of train spectra: {len(train_spectra)}")

    save_pickled_file(
        validation_spectra,
        os.path.join(
            data_directory,
            "training_and_validation_split",
            f"{ionization_mode}_validation_spectra.pickle",
        ),
    )
    save_pickled_file(
        test_spectra,
        os.path.join(
            data_directory,
            "training_and_validation_split",
            f"{ionization_mode}_test_spectra.pickle",
        ),
    )
    save_pickled_file(
        train_spectra,
        os.path.join(
            data_directory,
            "training_and_validation_split",
            f"{ionization_mode}_training_spectra.pickle",
        ),
    )
