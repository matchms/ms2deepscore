import numba
import numpy as np
import torch
from ms2deepscore.MetadataFeatureGenerator import (MetadataVectorizer,
                                                   load_from_json)
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore


def tensorize_spectra(
    spectra,
    settings: SettingsMS2Deepscore,
    ):
    """Convert list of matchms Spectrum objects to pytorch peak and metadata tensors.
    """
    if len(settings.additional_metadata) == 0:
        metadata_tensors = torch.zeros((len(spectra), 0))
    else:
        feature_generators = load_from_json(settings.additional_metadata)
        metadata_vectorizer = MetadataVectorizer(additional_metadata=feature_generators)
        metadata_tensors = metadata_vectorizer.transform(spectra)

    binned_spectra = torch.zeros((len(spectra), settings.number_of_bins()))
    for i, spectrum in enumerate(spectra):
        binned_spectra[i, :] = torch.tensor(vectorize_spectrum(spectrum.peaks.mz, spectrum.peaks.intensities,
                                                               settings.min_mz,
                                                               settings.max_mz,
                                                               settings.mz_bin_width,
                                                               settings.intensity_scaling
                                                               ))
    return binned_spectra, metadata_tensors


@numba.jit(nopython=True)
def vectorize_spectrum(mz_array, intensities_array, min_mz, max_mz, mz_bin_width, intensity_scaling):
    """Fast function to convert mz and intensity arrays into dense spectrum vector."""
    num_bins = int((max_mz - min_mz) / mz_bin_width)
    vector = np.zeros((num_bins))
    for mz, intensity in zip(mz_array, intensities_array):
        if min_mz <= mz < max_mz:
            bin_index = int((mz - min_mz) / mz_bin_width)
            # Take max intensity peak per bin
            vector[bin_index] = max(vector[bin_index], intensity ** intensity_scaling)
            # Alternative: Sum all intensties for all peaks in each bin
            # vector[bin_index] += intensity ** intensity_scaling
    return vector