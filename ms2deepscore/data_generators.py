""" Data generators for training/inference with MS2DeepScore model.
"""
import pickle
from typing import List
import numba
import numpy as np
import torch
from matchms import Spectrum
from ms2deepscore.MetadataFeatureGenerator import (MetadataVectorizer,
                                                   load_from_json)
from ms2deepscore.train_new_model.spectrum_pair_selection import (
    SelectedCompoundPairs, select_compound_pairs_wrapper)
from .SettingsMS2Deepscore import GeneratorSettings


class TensorizationSettings:
    """Stores the settings for tensorizing Spectra"""
    def __init__(self,
                 **settings):
        self.min_mz = 10
        self.max_mz = 1000
        self.mz_bin_width = 0.1
        self.intensity_scaling = 0.5
        self.additional_metadata = []
        if settings:
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown setting: {key}")
        self.num_bins = int((self.max_mz - self.min_mz) / self.mz_bin_width)

    def get_dict(self):
        return {"min_mz": self.min_mz,
                "max_mz": self.max_mz,
                "mz_bin_width": self.mz_bin_width,
                "intensity_scaling": self.intensity_scaling,
                "additional_metadata": self.additional_metadata}


def compute_validation_set(spectrums, tensorization_settings, generator_settings):
    """Since the pair selection is a highly randomized process,
    this functions will generate a validation data generator that can be stored
    using pickle. This is meant for consistent use throughout a larger project,
    e.g. for consistent hyperparameter optimization.
    """
    if not generator_settings.use_fixed_set:
        print("Expected use_fixed_set to be True (was changed to True)")
        generator_settings.use_fixed_set = True
    if not generator_settings.random_seed:
        print("Expected random seed (was set to 0)")
        generator_settings.random_seed = 0

    scp_val, _ = select_compound_pairs_wrapper(spectrums, generator_settings, shuffling=False)

    val_generator = DataGeneratorPytorch(
        spectrums=spectrums,
        tensorization_settings=tensorization_settings,
        selected_compound_pairs=scp_val,
        **generator_settings.__dict__,
    )
    # Collect batches
    print(f"Collecting {len(val_generator)} batches of size {val_generator.settings.batch_size}")
    for _ in val_generator:
        pass

    # Remove spectrums (no longer needed)
    del val_generator.spectrums

    return val_generator


def write_to_pickle(generator, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(generator, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_generator_from_pickle(filepath):
    with open(filepath, "rb") as f:
        generator = pickle.load(f)
    return generator


class DataGeneratorPytorch:
    """Generates data for training a siamese Keras model.

    This class provides a data generator specifically
    designed for training a siamese Keras model with a curated set of compound pairs.
    It uses pre-selected compound pairs, allowing more control over the training process,
    particularly in scenarios where certain compound pairs are of specific interest or
    have higher significance in the training dataset.
    """
    def __init__(self, spectrums: List[Spectrum],
                 selected_compound_pairs: SelectedCompoundPairs,
                 tensorization_settings: TensorizationSettings,
                 **settings):
        """Generates data for training a siamese Keras model.

        Parameters
        ----------
        spectrums
            List of matchms Spectrum objects.
        selected_compound_pairs
            SelectedCompoundPairs object which contains selected compounds pairs and the
            respective similarity scores.
        min_mz
            Lower bound for m/z values to consider.
        max_mz
            Upper bound for m/z values to consider.
        mz_bin_width
            Bin width for m/z sampling.
        intensity_scaling
            To put more attention on small and medium intensity peaks, peak intensities are
             scaled by intensity to the power of intensity_scaling.
        metadata_vectorizer
            Add the specific MetadataVectorizer object for your data if the model should contain specific
            metadata entries as input. Default is set to None which means this will be ignored.
        settings
            The available settings can be found in GeneratorSettings
        """
        self.current_index = 0
        self.spectrums = spectrums

        # Collect all inchikeys
        self.spectrum_inchikeys = np.array([s.get("inchikey")[:14] for s in self.spectrums])

        # Set all other settings to input (or otherwise to defaults):
        self.settings = GeneratorSettings(settings)
        self.tensorization_settings = tensorization_settings

        # Initialize random number generator
        if self.settings.use_fixed_set:
            if selected_compound_pairs.shuffling:
                raise ValueError("The generator cannot run reproducibly when shuffling is on for `SelectedCompoundPairs`.")
            if self.settings.random_seed is None:
                self.settings.random_seed = 0
        self.rng = np.random.default_rng(self.settings.random_seed)

        unique_inchikeys = np.unique(self.spectrum_inchikeys)
        if len(unique_inchikeys) < self.settings.batch_size:
            raise ValueError("The number of unique inchikeys must be larger than the batch size.")
        self.fixed_set = {}

        self.selected_compound_pairs = selected_compound_pairs
        self.on_epoch_end()

    def __len__(self):
        return int(self.settings.num_turns)\
            * int(np.ceil(len(self.selected_compound_pairs.scores) / self.settings.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < self.__len__():
            batch = self.__getitem__(self.current_index)
            self.current_index += 1
            return batch
        self.current_index = 0  # make generator executable again
        self.on_epoch_end()
        raise StopIteration

    def _spectrum_pair_generator(self, batch_index: int):
        """Use the provided SelectedCompoundPairs object to pick pairs."""
        batch_size = self.settings.batch_size
        indexes = self.indexes[batch_index * batch_size:(batch_index + 1) * batch_size]
        for index in indexes:
            inchikey1 = self.selected_compound_pairs.idx_to_inchikey[index]
            score, inchikey2 = self.selected_compound_pairs.next_pair_for_inchikey(inchikey1)
            spectrum1 = self._get_spectrum_with_inchikey(inchikey1)
            spectrum2 = self._get_spectrum_with_inchikey(inchikey2)
            yield (spectrum1, spectrum2, score)

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.tile(np.arange(len(self.selected_compound_pairs.scores)), int(self.settings.num_turns))
        if self.settings.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        if self.settings.use_fixed_set and batch_index in self.fixed_set:
            return self.fixed_set[batch_index]
        if self.settings.random_seed is not None and batch_index == 0:
            self.rng = np.random.default_rng(self.settings.random_seed)
        spectrum_pairs = self._spectrum_pair_generator(batch_index)
        spectra_1, spectra_2, meta_1, meta_2, targets = self._tensorize_all(spectrum_pairs)

        if self.settings.use_fixed_set:
            # Store batches for later epochs
            self.fixed_set[batch_index] = (spectra_1, spectra_2, meta_1, meta_2, targets)
        else:
            spectra_1 = self._data_augmentation(spectra_1)
            spectra_2 = self._data_augmentation(spectra_2)
        return spectra_1, spectra_2, meta_1, meta_2, targets

    def _tensorize_all(self, spectrum_pairs):
        spectra_1 = []
        spectra_2 = []
        targets = []
        for pair in spectrum_pairs:
            spectra_1.append(pair[0])
            spectra_2.append(pair[1])
            targets.append(pair[2])

        binned_spectra_1, metadata_1 = tensorize_spectra(
            spectra_1,
            self.tensorization_settings
            )
        binned_spectra_2, metadata_2 = tensorize_spectra(
            spectra_2,
            self.tensorization_settings
            )
        return binned_spectra_1, binned_spectra_2, metadata_1, metadata_2, torch.tensor(targets, dtype=torch.float32)
    
    def _get_spectrum_with_inchikey(self, inchikey: str) -> Spectrum:
        """
        Get a random spectrum matching the `inchikey` argument.

        NB: A compound (identified by an
        inchikey) can have multiple measured spectrums in a binned spectrum dataset.
        """
        matching_spectrum_id = np.where(self.spectrum_inchikeys == inchikey)[0]
        if len(matching_spectrum_id) <= 0:
            raise ValueError("No matching inchikey found (note: expected first 14 characters)")
        return self.spectrums[self.rng.choice(matching_spectrum_id)]

    def _data_augmentation(self, spectra_tensors):
        for i in range(spectra_tensors.shape[0]):
            spectra_tensors[i, :] = self._data_augmentation_spectrum(spectra_tensors[i, :])
        return spectra_tensors
    
    def _data_augmentation_spectrum(self, spectrum_tensor):
        """Data augmentation.

        Parameters
        ----------
        spectrum_tensor
            Spectrum in Pytorch tensor form.
        """
        # Augmentation 1: peak removal (peaks < augment_removal_max)
        if self.settings.augment_removal_max or self.settings.augment_removal_intensity:
            # TODO: Factor out function with documentation + example?
            
            indices_select = torch.where((spectrum_tensor > 0) 
                                        & (spectrum_tensor < self.settings.augment_removal_max))[0]
            removal_part = self.rng.random(1) * self.settings.augment_removal_max
            indices = self.rng.choice(indices_select, int(np.ceil((1 - removal_part)*len(indices_select))))
            if len(indices) > 0:
                spectrum_tensor[indices] = 0

        # Augmentation 2: Change peak intensities
        if self.settings.augment_intensity:
            # TODO: Factor out function with documentation + example?
            spectrum_tensor = spectrum_tensor * (1 - self.settings.augment_intensity * 2 * (torch.rand(spectrum_tensor.shape) - 0.5))

        # Augmentation 3: Peak addition
        if self.settings.augment_noise_max and self.settings.augment_noise_max > 0:
            indices_select = torch.where(spectrum_tensor == 0)[0]
            if len(indices_select) > self.settings.augment_noise_max:
                indices_noise = self.rng.choice(indices_select,
                                                 self.rng.integers(0, self.settings.augment_noise_max),
                                                 replace=False,
                                                )
            spectrum_tensor[indices_noise] = self.settings.augment_noise_intensity * torch.rand(len(indices_noise))
        return spectrum_tensor


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
