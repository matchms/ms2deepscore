import os
import string
from pathlib import Path

import numpy as np
from matchms import Spectrum
from matchms.importing import load_from_mgf
from ms2deepscore.models import load_model
from ms2deepscore.MS2DeepScore import MS2DeepScore


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def pesticides_test_spectra():
    return list(load_from_mgf(os.path.join(TEST_RESOURCES_PATH, "pesticides_processed.mgf")))


def ms2deepscore_model_file_name():
    return os.path.join(TEST_RESOURCES_PATH, "testmodel_v1.pt")


def siamese_spectral_model():
    return load_model(ms2deepscore_model_file_name())


def ms2deepscore_object():
    return MS2DeepScore(load_model(ms2deepscore_model_file_name()))


def create_test_spectra(num_of_unique_inchikeys: int,
                        num_of_spectra_per_inchikey: int = 1):
    if num_of_unique_inchikeys > 26:
        raise ValueError("The max number of unique inchikeys is 26, because that is the number of available letters")

    intens = [0.1, 1]
    spectrums = []
    letters = list(string.ascii_uppercase[:num_of_unique_inchikeys])

    # 26 valid and fairly similar substituents
    substituents = [
        "C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
        "O", "OC", "OCC", "OCCC",
        "N", "NC", "NCC",
        "F", "Cl", "Br", "I",
        "C(F)(F)F", "C#N", "C(=O)O", "C(=O)OC",
        "S(=O)(=O)N", "S(=O)(=O)C",
        "c1ccccc1", "c1ccncc1", "COC"
    ]

    for i, letter in enumerate(letters):
        dummy_inchikey = f"{14 * letter}-{10 * letter}-N"
        substituent = substituents[i]

        # Shared aromatic scaffold with varying para substituent
        smiles = f"CC(=O)Oc1ccc({substituent})cc1"

        for j in range(num_of_spectra_per_inchikey):
            spectrums.append(
                Spectrum(
                    mz=np.array([100 + (i + 1) * 0.2, 500 + j * 0.2]),
                    intensities=np.array(intens),
                    metadata={
                        "precursor_mz": 111.1,
                        "inchikey": dummy_inchikey,
                        "compound_name": letter,
                        "smiles": smiles
                    }
                )
            )
    return spectrums
