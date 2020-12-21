from typing import List
import numpy as np
from tqdm import tqdm
from ms2deepscore.spectrum_binning_fixed import create_peak_list_fixed
from ms2deepscore.spectrum_binning_fixed import set_d_bins_fixed
from ms2deepscore.spectrum_binning_fixed import unique_peaks_fixed
from ms2deepscore.utils import create_peak_dict


class SpectrumBinner:
    """Create binned spectrum data and keep track of parameters.

    TODO: add description --> here: fixed bins!
    """
    def __init__(self, number_of_bins: int,
                 mz_max: float = 1000.0, mz_min: float = 10.0,
                 peak_scaling: float = 0.5, allowed_missing_percentage: float = 0.0):
        """

        Parameters
        ----------
        number_of_bins
            Number if bins to represent spectrum.
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
        assert mz_max > mz_min, "mz_max must be > mz>min"
        self.mz_max = mz_max
        self.mz_min = mz_min
        self.d_bins = set_d_bins_fixed(number_of_bins, mz_min=mz_min, mz_max=mz_max)
        self.peak_scaling = peak_scaling
        self.allowed_missing_percentage = allowed_missing_percentage
        self.peak_to_position = None
        self.known_bins = None
        self.generator_args = {}
        self.spectrums_binned = None
        self.inchikeys_all = None

    def collect_binned_spectrums(self, spectrums: List[Spectrum], progress_bar=True):
        """Transforms the input *spectrums* into binned spectrums as needed for
        MS2DeepScore.
        
        Includes creating a 'vocabulary' of bins that have peaks in spectrums,
        which is stored in SpectrumBinner.known_bins. 
        Creates binned spectrums from input spectrums and stores them in
        SpectrumBinner.spectrums_binned.

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
        self.spectrums_binned = self._create_binned_spectrums(spectrums, progress_bar)

        # Collect inchikeys
        self._collect_inchikeys(spectrums)

    def _create_binned_spectrums(self, input_spectrums: List[Spectrum],
                                 progress_bar=True) -> List[dict]:
        """Create binned spectrums from input spectrums.

        Parameters
        ----------
        input_spectrums
            List of spectrums.
        progress_bar
            Show progress bar if set to True. Default is True.

        Returns: List[dict]
            List of binned spectrums created from input_spectrums.
        """
        spectrums_binned, missing_fractions = create_peak_list_fixed(input_spectrums,
                                                                     self.peak_to_position,
                                                                     self.d_bins, mz_min=self.mz_min,
                                                                     peak_scaling=self.peak_scaling)
        spectrums_binned_dicts = []
        for i, spec in enumerate(tqdm(spectrums_binned, disable=(not progress_bar))):
            assert 100*missing_fractions[i] <= self.allowed_missing_percentage, \
                f"{100*missing_fractions[i]:.2f} of weighted spectrum is unknown to the model."
            spectrums_binned_dicts.append(create_peak_dict(spec))
        return spectrums_binned_dicts

    def _collect_inchikeys(self, spectrums):
        """Read inchikeys from spectrums and create inchkeys array.

        Parameters
        ----------
        spectrums
            List of spectrums.
        """
        inchikeys_list = []
        for s in spectrums:
            inchikeys_list.append(s.get("inchikey"))

        self.inchikeys_all = np.array(inchikeys_list)

    def set_generator_parameters(self, **settings):
        """Set parameter for data generator. Use below listed defaults unless other
        input is provided.

        TODO: Move this part to the generator (not sure)

        Parameters
        ----------
        batch_size
            Number of pairs per batch. Default=32.
        num_turns
            Number of pairs for each InChiKey during each epoch. Default=1
        shuffle
            Set to True to shuffle IDs every epoch. Default=True
        ignore_equal_pairs
            Set to True to ignore pairs of two identical spectra. Default=True
        same_prob_bins
            List of tuples that define ranges of the true label to be trained with
            equal frequencies. Default is set to [(0, 0.5), (0.5, 1)], which means
            that pairs with scores <=0.5 will be picked as often as pairs with scores
            > 0.5.
        augment_peak_removal_max
            Maximum fraction of peaks (if intensity < below augment_peak_removal_intensity)
            to be removed randomly. Default is set to 0.2, which means that between
            0 and 20% of all peaks with intensities < augment_peak_removal_intensity
            will be removed.
        augment_peak_removal_intensity
            Specifying that only peaks with intensities < max_intensity will be removed.
        augment_intensity
            Change peak intensities by a random number between 0 and augment_intensity.
            Default=0.1, which means that intensities are multiplied by 1+- a random
            number within [0, 0.1].
        """
        defaults = dict(
            batch_size=32,
            num_turns=1,
            ignore_equal_pairs=True,
            shuffle=True,
            same_prob_bins=[(0, 0.5), (0.5, 1)],
            augment_peak_removal_max= 0.3,
            augment_peak_removal_intensity=0.2,
            augment_intensity=0.4,
        )

        # Set default parameters or replace by **settings input
        for key in defaults:
            if key in settings:
                print("The value for {} is set from {} (default) to {}".format(key, defaults[key],
                                                                              settings[key]))
            elif key in self.generator_args and self.generator_args[key] != defaults[key]:
                print("Previously set value for {} of {} is reset to default ({})".format(key, self.generator_args[key],
                                                                                          defaults[key]))
                settings[key] = defaults[key]
            else:
                settings[key] = defaults[key]
        assert 0.0 <= settings["augment_peak_removal_max"] <= 1.0, "Expected value within [0,1]"
        assert 0.0 <= settings["augment_peak_removal_intensity"] <= 1.0, "Expected value within [0,1]"
        self.generator_args = settings
