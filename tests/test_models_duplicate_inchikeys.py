import os
from pathlib import Path
import pytest
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import (DataGeneratorAllInchikeys,
                                          DataGeneratorAllSpectrums)
from tests.test_user_worfklow import (get_reference_scores,
                                      load_processed_spectrums)


def test_error_duplicate_inchikeys():
    """Test an expected error when duplicate inchikeys are given to DataGenerator"""
    ## Get test data ##
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()

    ## Create duplicate inchikeys ##
    sel = list(range(30)) + list(range(30))
    tanimoto_scores_df = tanimoto_scores_df.iloc[sel, sel]
    selected_inchikeys = tanimoto_scores_df.index[:60].unique()

    ## Subset spectra to selected inchikeys and bin
    spectrums = [s for s in spectrums if s.get("inchikey")[:14] in selected_inchikeys]
    spectrum_binner = SpectrumBinner(400, mz_min=10.0, mz_max=500.0, peak_scaling=0.5)
    binned_spectrums = spectrum_binner.fit_transform(spectrums)
    dimension = len(spectrum_binner.known_bins)

    ## Setup DataGenerator
    with pytest.raises(ValueError):
        DataGeneratorAllInchikeys(binned_spectrums=binned_spectrums,
                                  selected_inchikeys=selected_inchikeys,
                                  reference_scores_df=tanimoto_scores_df,
                                  spectrum_binner=spectrum_binner)

    with pytest.raises(ValueError):
        DataGeneratorAllSpectrums(binned_spectrums=binned_spectrums,
                                  reference_scores_df=tanimoto_scores_df,
                                  spectrum_binner=spectrum_binner)
