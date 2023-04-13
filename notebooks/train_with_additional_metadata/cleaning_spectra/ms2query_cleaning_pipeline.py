from typing import List, Tuple
import matchms.filtering as msfilters
from tqdm import tqdm
from matchms import Spectrum
from matchms.metadata_utils import is_valid_inchi, is_valid_inchikey, is_valid_smiles
from matchms.typing import SpectrumType
from matchms.logging_functions import set_matchms_logger_level


def clean_metadata(spectrum: Spectrum) -> SpectrumType:
    spectrum = msfilters.default_filters(spectrum)
    spectrum = msfilters.require_precursor_mz(spectrum)
    return spectrum


def normalize_and_filter_peaks(spectrum: Spectrum) -> SpectrumType:
    """Spectrum is normalized and filtered"""
    spectrum = msfilters.normalize_intensities(spectrum)
    spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
    spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
    spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
    spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)
    return spectrum


def harmonize_annotation(spectrum: Spectrum) -> SpectrumType:
    set_matchms_logger_level("CRITICAL")
    # Here, undefiend entries will be harmonized (instead of having a huge variation of None,"", "N/A" etc.)
    spectrum = msfilters.harmonize_undefined_inchikey(spectrum)
    spectrum = msfilters.harmonize_undefined_inchi(spectrum)
    spectrum = msfilters.harmonize_undefined_smiles(spectrum)

    # The repair_inchi_inchikey_smiles function will correct misplaced metadata
    # (e.g. inchikeys entered as inchi etc.) and harmonize the entry strings.
    spectrum = msfilters.repair_inchi_inchikey_smiles(spectrum)

    # Where possible (and necessary, i.e. missing): Convert between smiles, inchi, inchikey to complete metadata.
    # This is done using functions from rdkit.
    spectrum = msfilters.derive_inchi_from_smiles(spectrum)
    spectrum = msfilters.derive_smiles_from_inchi(spectrum)
    spectrum = msfilters.derive_inchikey_from_inchi(spectrum)
    return spectrum


def remove_wrong_ion_modes(spectra, ion_mode_to_keep):
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


def split_annotated_spectra(spectra: List[SpectrumType]) -> Tuple[List[Spectrum], List[Spectrum]]:
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


def normalize_and_filter_peaks_multiple_spectra(spectrum_list: List[SpectrumType],
                                                progress_bar: bool = False
                                                ) -> List[SpectrumType]:
    """Preprocesses all spectra and removes None values

    Args:
    ------
    spectrum_list:
        List of spectra that should be preprocessed.
    progress_bar:
        If true a progress bar will be shown.
    """
    for i, spectrum in enumerate(
            tqdm(spectrum_list,
                 desc="Preprocessing spectra",
                 disable=not progress_bar)):
        processed_spectrum = normalize_and_filter_peaks(spectrum)
        spectrum_list[i] = processed_spectrum

    # Remove None values
    return [spectrum for spectrum in spectrum_list if spectrum]


def clean_annotation_and_peaks(spectra):
    spectra = [harmonize_annotation(s) for s in tqdm(spectra, desc="Harmonizing annotations")]
    spectra = normalize_and_filter_peaks_multiple_spectra(spectra, progress_bar=True)
    annotated_spectra, unannotated_spectra = split_annotated_spectra(spectra)
    return annotated_spectra

def clean_and_split_spectra_on_mode(spectra: List[Spectrum]) -> Tuple[List[Spectrum], List[Spectrum]]:
    cleaned_spectra = [clean_metadata(s) for s in tqdm(spectra, desc="Cleaning metadata")]
    positive_spectra = remove_wrong_ion_modes(cleaned_spectra, "positive")
    negative_spectra = remove_wrong_ion_modes(cleaned_spectra, "negative")
    return positive_spectra, negative_spectra


if __name__ == "__main__":
    import pickle
    import os

    def save_pickled_file(obj, filename: str):
        assert not os.path.exists(filename), "File already exists"
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def load_pickled_file(filename: str):
        with open(filename, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object

    # spectra = load_pickled_file("../../../../data/all_spectra_230201_matchms_manual_pubchem.pickle")
    # positive_mode_spectra, negative_mode_spectra = clean_and_split_spectra_on_mode(spectra)
    # save_pickled_file(negative_mode_spectra, "../../../../data/cleaned_spectra/negative_mode_spectra.pickle")
    positive_mode_spectra =load_pickled_file("../../../../data/cleaned_spectra/positive_mode_spectra.pickle")

    # negative_annotated_spectra = clean_annotation_and_peaks(negative_mode_spectra)
    # save_pickled_file(negative_annotated_spectra, "../../../../data/cleaned_spectra/negative_annotated_spectra.pickle")

    positive_annotated_spectra = clean_annotation_and_peaks(positive_mode_spectra)
    save_pickled_file(positive_annotated_spectra, "../../../../data/cleaned_spectra/positive_annotated_spectra.pickle")
