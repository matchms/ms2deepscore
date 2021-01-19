from typing import List
import numpy as np
from tqdm import tqdm
from matchms.typing import SpectrumType
from ms2deepscore import BinnedSpectrum
from ms2deepscore.spectrum_binning_fixed import create_peak_list_fixed
from ms2deepscore.spectrum_binning_fixed import set_d_bins_fixed
from ms2deepscore.spectrum_binning_fixed import unique_peaks_fixed
from ms2deepscore.utils import create_peak_dict


class SpectrumBinner:
    """Create binned spectrum data and keep track of parameters.

    Converts input spectrums into :class:`~ms2deepscore.BinnedSpectrum` objects.
    Binning is here done using a fixed bin width defined by the *number_of_bins*
    as well as the range set by *mz_min* and *mz_max*.
    """
    def __init__(self, number_of_bins: int,
                 mz_max: float = 1000.0, mz_min: float = 10.0,
                 peak_scaling: float = 0.5, allowed_missing_percentage: float = 0.0):
        """

        Parameters
        ----------
        number_of_bins
            Number of bins to represent spectrum.
        mz_max
            Upper bound of m/z to include in binned spectrum. Default is 1000.0.
        mz_min
            Lower bound of m/z to include in binned spectrum. Default is 10.0.
        peak_scaling
            Scale all peak intensities by power pf peak_scaling. Default is 0.5.
        allowed_missing_percentage:
            Set the maximum allowed percentage of the spectrum that may be unknown
            from the input model. This is measured as percentage of the weighted, unknown
            binned peaks compared to all peaks of the spectrum. Default is 0, which
            means no unknown binned peaks are allowed.
        """
        self.number_of_bins = number_of_bins
        assert mz_max > mz_min, "mz_max must be > mz_min"
        self.mz_max = mz_max
        self.mz_min = mz_min
        self.d_bins = set_d_bins_fixed(number_of_bins, mz_min=mz_min, mz_max=mz_max)
        self.peak_scaling = peak_scaling
        self.allowed_missing_percentage = allowed_missing_percentage
        self.peak_to_position = None
        self.known_bins = None
        self.binned_spectrums = None

    def fit_transform(self, spectrums: List[SpectrumType], progress_bar=True):
        """Transforms the input *spectrums* into binned spectrums as needed for
        MS2DeepScore.

        Includes creating a 'vocabulary' of bins that have peaks in spectrums,
        which is stored in SpectrumBinner.known_bins.
        Creates binned spectrums from input spectrums and stores them in
        SpectrumBinner.binned_spectrums.

        Parameters
        ----------
        spectrums
            List of spectrums.
        progress_bar
            Show progress bar if set to True. Default is True.
        """
        print("Collect spectrum peaks...")
        peak_to_position, known_bins = unique_peaks_fixed(spectrums, self.d_bins, self.mz_min)
        print(f"Calculated embedding dimension of {len(known_bins)}.")
        self.peak_to_position = peak_to_position
        self.known_bins = known_bins

        print("Convert spectrums to binned spectrums...")
        return self.transform(spectrums, progress_bar)

    def transform(self, input_spectrums: List[SpectrumType],
                  progress_bar=True) -> List[BinnedSpectrum]:
        """Create binned spectrums from input spectrums.

        Parameters
        ----------
        input_spectrums
            List of spectrums.
        progress_bar
            Show progress bar if set to True. Default is True.

        Returns:
            List of binned spectrums created from input_spectrums.
        """
        peak_lists, missing_fractions = create_peak_list_fixed(input_spectrums,
                                                               self.peak_to_position,
                                                               self.d_bins, mz_min=self.mz_min,
                                                               peak_scaling=self.peak_scaling)
        spectrums_binned = []
        for i, peak_list in enumerate(tqdm(peak_lists, disable=(not progress_bar))):
            assert 100*missing_fractions[i] <= self.allowed_missing_percentage, \
                f"{100*missing_fractions[i]:.2f} of weighted spectrum is unknown to the model."
            spectrum = BinnedSpectrum(binned_peaks=create_peak_dict(peak_list),
                                      metadata={"inchikey": input_spectrums[i].get("inchikey")})
            spectrums_binned.append(spectrum)
        return spectrums_binned
