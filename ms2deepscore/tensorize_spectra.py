import numba
import numpy as np
import torch

from ms2deepscore.MetadataFeatureGenerator import load_from_json, MetadataVectorizer
from ms2deepscore.SettingsMS2Deepscore import TensorizationSettings


def tensorize_spectra(
    spectra,
    tensorization_settings: TensorizationSettings,
    ):
    """Convert list of matchms Spectrum objects to pytorch peak and metadata tensors.
    """
    if len(tensorization_settings.additional_metadata) == 0:
        metadata_tensors = torch.zeros((len(spectra), 0))
    else:
        feature_generators = load_from_json(tensorization_settings.additional_metadata)
        metadata_vectorizer = MetadataVectorizer(additional_metadata=feature_generators)
        metadata_tensors = metadata_vectorizer.transform(spectra)

    binned_spectra = torch.zeros((len(spectra), tensorization_settings.num_bins))
    for i, spectrum in enumerate(spectra):
        binned_spectra[i, :] = torch.tensor(vectorize_spectrum(spectrum.peaks.mz, spectrum.peaks.intensities,
                                                               tensorization_settings.min_mz,
                                                               tensorization_settings.max_mz,
                                                               tensorization_settings.mz_bin_width,
                                                               tensorization_settings.intensity_scaling
                                                               ))
    return binned_spectra, metadata_tensors


@numba.jit(nopython=True)
def vectorize_spectrum(mz_array, intensities_array, min_mz, max_mz, mz_bin_width, intensity_scaling):
    """Fast function to convert mz and intensity arrays into dense spectrum vector."""
    # pylint: disable=too-many-arguments
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