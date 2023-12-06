from typing import List, Tuple
from matchms import Spectrum
from tqdm import tqdm


def split_by_ionmode(spectra:List[Spectrum]) -> Tuple[List[Spectrum], List[Spectrum]]:
    """Splits spectra into list of positive ionmode and list of negative ionmode spectra.

    Removes spectra without correct ionmode metadata entry.
    """
    pos_spectra = []
    neg_spectra = []
    spectra_removed = 0
    for spectrum in tqdm(spectra,
                         desc="Splitting pos and neg mode spectra"):
        if spectrum is not None:
            ionmode = spectrum.get("ionmode")
            if ionmode == "positive":
                pos_spectra.append(spectrum)
            elif ionmode == "negative":
                neg_spectra.append(spectrum)
            else:
                spectra_removed += 1
    print(f"The spectra, are split in {len(pos_spectra)} positive spectra "
          f"and {len(neg_spectra)} negative mode spectra. {spectra_removed} were removed")
    return pos_spectra, neg_spectra
