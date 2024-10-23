from typing import List
from matchms.Spectrum import Spectrum
from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import calculate_tanimoto_scores_unique_inchikey


def get_tanimoto_score_between_spectra(spectra_1: List[Spectrum],
                                       spectra_2: List[Spectrum],
                                       fingerprint_type="daylight",
                                       nbits=2048):
    """Gets the tanimoto scores between two list of spectra

    It is optimized by calculating the tanimoto scores only between unique fingerprints/smiles.
    The tanimoto scores are derived after.

    """
    tanimoto_df = calculate_tanimoto_scores_unique_inchikey(spectra_1, spectra_2,
                                                            fingerprint_type,
                                                            nbits)
    inchikeys_1 = [spectrum.get("inchikey")[:14] for spectrum in spectra_1]
    inchikeys_2 = [spectrum.get("inchikey")[:14] for spectrum in spectra_2]
    tanimoto_scores = tanimoto_df.loc[inchikeys_1, inchikeys_2].values
    return tanimoto_scores
