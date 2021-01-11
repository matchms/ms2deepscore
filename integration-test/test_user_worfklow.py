import os
import numpy as np
import pytest
from matchms import calculate_scores
from matchms.filtering import add_parent_mass
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.importing import load_from_mgf
from matchms import Spectrum
from ms2deepscore import SpectrumBinner


def load_process_spectrums():
    """l"L"oad and process spectrums from mgf file."""
    def apply_my_filters(s):
        s = default_filters(s)
        s = add_parent_mass(s)
        s = normalize_intensities(s)
        s = select_by_relative_intensity(s, intensity_from=0.0, intensity_to=1.0)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = require_minimum_number_of_peaks(s, n_required=5)
        return s

    #module_root = os.path.join(os.path.dirname(__file__), "..")
    module_root = 'C:\\OneDrive - Netherlands eScience Center\\Project_Wageningen_iOMEGA\\ms2deepscore\\'
    spectrums_file = os.path.join(module_root, "integration-test", "pesticides.mgf")

    # apply my filters to the data
    spectrums = [apply_my_filters(s) for s in load_from_mgf(spectrums_file)]

    # omit spectrums that didn't qualify for analysis
    spectrums = [s for s in spectrums if s is not None]
    return spectrums


def test_user_workflow():
    """Test if a typical user workflow."""

    # def test_user_workflow():

    # Load and process spectrums
    spectrums = load_process_spectrums()

    # Create binned spectrums
    ms2ds_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
    spectrums_binned = ms2ds_binner.fit_transform(spectrums)

    assert len(ms2ds_binner.known_bins) == 543, "Expected differnt number of known binned peaks"

    # train model

    # or: load model

    # calculate similarities
