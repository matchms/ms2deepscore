from ms2deepscore.spectrum_binning_fixed import set_d_bins_fixed
from ms2deepscore.spectrum_binning_fixed import unique_peaks_fixed


class MS2DeepScore:
    """Create MS2DeepScore model.

    TODO: add description --> here: fixed bins!
    """
    def __init__(self, number_of_bins: int,
                 mz_max: float = 1000.0, mz_min: float = 10.0):
        """

        Parameters
        ----------
        number_of_bins
            Number if bins to represent spectrum.
        mz_max
            Upper bound of m/z to include in binned spectrum. Default is 1000.0.
        mz_max
            Lower bound of m/z to include in binned spectrum. Default is 10.0.
        """
        self.number_of_bins = number_of_bins
        assert mz_max > mz_min, "mz_max must be > mz>min"
        self.mz_max = mz_max
        self.mz_min = mz_min
        self.d_bins = set_d_bins_fixed(number_of_bins, mz_min=mz_min, mz_max=mz_max)
        self.peak_to_position = None
        self.known_bins = None
        self.model = None
        self.training_args = None
        self.spectrums_binned = None

    def create_binned_spectrums(self, spectrums: list):
        """Create 'vocabulary' of bins that have peaks in spectrums.
        Derive binned spectrums from spectrums.   

        Parameters
        ----------
        spectrums
            List of spectrums.
        """
        print("Collect spectrum peaks...")
        peak_to_position, known_bins = unique_peaks_fixed(spectrums, self.d_bins, self.mz_min)
        print(f"Calculated embedding dimension of {len(known_bins)}.")
        self.peak_to_position = peak_to_position
        self.known_bins = known_bins

        print("Convert spectrums to binned spectrums...")
        spectrums_binned = create_peak_list_fixed(spectrums, self.known_bins,
                                                  self.d_bins, mz_min=self.mz_min)
        self.spectrums_binned = [create_peak_dict(spec) for spec in spectrums_binned]

    def set_training_parameters(self, **settings):
        """Set parameter defaults"""
        defaults = dict(
            batch_size=25,
            num_turns=1,
            peak_scaling=0.5,
            ignore_equal_pairs=True,
            shuffle=True,
            same_prob_bins=same_prob_bins,
            augment_peak_removal_max= 0.3,
            augment_peak_removal_intensity=0.2,
            augment_intensity=0.4)
        )

        # Set default parameters or replace by **settings input
        for key in defaults:
            if key in settings:
                print("The value of {} is set from {} (default) to {}".format(key, defaults[key],
                                                                              settings[key]))
            else:
                settings[key] = defaults[key]
        self.training_args = settings
