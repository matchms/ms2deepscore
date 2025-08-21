import numpy as np
import torch

from ms2deepscore import SettingsMS2Deepscore


def data_augmentation(spectra_tensors,
                      model_settings: SettingsMS2Deepscore,
                      random_number_generator):
    for i in range(spectra_tensors.shape[0]):
        spectra_tensors[i, :] = data_augmentation_spectrum(spectra_tensors[i, :],
                                                           model_settings,
                                                           random_number_generator)
    return spectra_tensors


def data_augmentation_spectrum(spectrum_tensor,
                               model_settings: SettingsMS2Deepscore,
                               random_number_generator):
    """Data augmentation.

    Parameters
    ----------
    spectrum_tensor
        Spectrum in Pytorch tensor form.
    """
    # Augmentation 1: peak removal (peaks < augment_removal_max)
    peak_removal_for_data_augmentation(spectrum_tensor, model_settings.augment_removal_max,
                                       model_settings.augment_removal_intensity, random_number_generator)

    # Augmentation 2: Change peak intensities
    if model_settings.augment_intensity:
        spectrum_tensor = change_peak_intensity(spectrum_tensor, model_settings)

    peak_addition_for_data_augmentation(spectrum_tensor, model_settings, random_number_generator)
    return spectrum_tensor

def peak_removal_for_data_augmentation(spectrum_tensor, augment_removal_max,
                                       augment_removal_intensity, random_number_generator):
    """Removes small peaks at random for data augmentation.

    Parameters
    spectrum_tensor:
        Tensorized spectrum
    augment_removal_max
        Maximum fraction of peaks (if intensity < below augment_removal_intensity)
        to be removed randomly. Default is set to 0.2, which means that between
        0 and 20% of all peaks with intensities < augment_removal_intensity
        will be removed.
    augment_removal_intensity
        Specifying that only peaks with intensities < max_intensity will be removed.
    random_number_generator
        Random number generator used to generate random numbers. Can be generated with np.random.default_rng(42)
    """
    if augment_removal_max or augment_removal_intensity:
        bin_indices_below_removal_intensity = torch.where((spectrum_tensor > 0)
                                     & (spectrum_tensor < augment_removal_intensity))[0]
        fraction_of_noise_to_remove = random_number_generator.random(1) * augment_removal_max
        number_of_peaks_to_remove = int(np.ceil((1 - fraction_of_noise_to_remove) * len(bin_indices_below_removal_intensity)))
        indices = random_number_generator.choice(bin_indices_below_removal_intensity, number_of_peaks_to_remove)
        if len(indices) > 0:
            spectrum_tensor[indices] = 0

def change_peak_intensity(spectrum_tensor, model_settings):
    return spectrum_tensor * (1 - model_settings.augment_intensity * 2 * (torch.rand(spectrum_tensor.shape) - 0.5))

def peak_addition_for_data_augmentation(spectrum_tensor, model_settings, random_number_generator):
    if model_settings.augment_noise_max and model_settings.augment_noise_max > 0:
        indices_select = torch.where(spectrum_tensor == 0)[0]
        if len(indices_select) > model_settings.augment_noise_max:
            indices_noise = random_number_generator.choice(
                indices_select,
                random_number_generator.integers(0, model_settings.augment_noise_max), replace=False,)
        spectrum_tensor[indices_noise] = model_settings.augment_noise_intensity * torch.rand(len(indices_noise))