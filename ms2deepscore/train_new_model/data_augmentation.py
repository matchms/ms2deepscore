import numpy as np
import torch
from torch import where

from ms2deepscore import SettingsMS2Deepscore


def data_augmentation(spectra_tensors, model_settings: SettingsMS2Deepscore, random_number_generator):
    for i in range(spectra_tensors.shape[0]):
        spectra_tensors[i, :] = data_augmentation_spectrum(
            spectra_tensors[i, :], model_settings, random_number_generator
        )
    return spectra_tensors


def data_augmentation_spectrum(spectrum_tensor, model_settings: SettingsMS2Deepscore, random_number_generator):
    """Apply reproducible peak-removal, intensity, and noise augmentation.
    
    Parameters
    ----------
    spectrum_tensor : torch.Tensor
        A 1D tensor representing the spectrum to be augmented.
    model_settings : SettingsMS2Deepscore
        Settings object containing augmentation parameters.
    random_number_generator : np.random.Generator
        A random number generator for reproducibility.
    """
    peak_removal_for_data_augmentation(
        spectrum_tensor,
        model_settings.augment_removal_max,
        model_settings.augment_removal_intensity,
        random_number_generator,
    )
    change_peak_intensity_for_data_augmentation(
        spectrum_tensor,
        model_settings.augment_intensity,
        random_number_generator,
    )
    peak_addition_for_data_augmentation(
        spectrum_tensor,
        model_settings.augment_noise_max,
        model_settings.augment_noise_intensity,
        random_number_generator,
    )
    return spectrum_tensor


def peak_removal_for_data_augmentation(
    spectrum_tensor, augment_removal_max, augment_removal_intensity, random_number_generator
):
    """Remove up to ``augment_removal_max`` of eligible low-intensity peaks.

    Parameters
    ----------
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
        Random number generator used to generate random numbers. 
    """
    if augment_removal_max <= 0:
        return

    candidate_indices = where(
        (spectrum_tensor > 0) & (spectrum_tensor < augment_removal_intensity)
    )[0]
    if len(candidate_indices) == 0:
        return

    fraction_to_remove = random_number_generator.random() * augment_removal_max
    number_of_peaks_to_remove = int(fraction_to_remove * len(candidate_indices))
    if number_of_peaks_to_remove == 0:
        return

    indices = random_number_generator.choice(
        candidate_indices.cpu().numpy(), number_of_peaks_to_remove, replace=False
    )
    spectrum_tensor[torch.as_tensor(indices, device=spectrum_tensor.device)] = 0


def change_peak_intensity_for_data_augmentation(
    spectrum_tensor, augment_intensity, random_number_generator=None
):
    if random_number_generator is None:
        random_number_generator = np.random.default_rng()
    if augment_intensity:
        factors = random_number_generator.uniform(
            1 - augment_intensity,
            1 + augment_intensity,
            size=tuple(spectrum_tensor.shape),
        )
        spectrum_tensor.mul_(
            torch.as_tensor(
                factors,
                dtype=spectrum_tensor.dtype,
                device=spectrum_tensor.device,
            )
        )


def peak_addition_for_data_augmentation(
    spectrum_tensor, augment_noise_max, augment_noise_intensity, random_number_generator
):
    """Add between 0 and ``augment_noise_max`` random noise peaks inclusive."""
    if not augment_noise_max or augment_noise_max <= 0:
        return

    bin_indices_zero = where(spectrum_tensor == 0)[0]
    number_of_noise_peaks_to_add = int(
        random_number_generator.integers(0, int(augment_noise_max) + 1)
    )
    if number_of_noise_peaks_to_add == 0 or len(bin_indices_zero) == 0:
        return

    available = bin_indices_zero.cpu().numpy()
    if len(available) > number_of_noise_peaks_to_add:
        selected = random_number_generator.choice(
            available, number_of_noise_peaks_to_add, replace=False
        )
    else:
        selected = available

    noise = random_number_generator.random(len(selected)) * augment_noise_intensity
    selected_tensor = torch.as_tensor(selected, device=spectrum_tensor.device)
    spectrum_tensor[selected_tensor] = torch.as_tensor(
        noise,
        dtype=spectrum_tensor.dtype,
        device=spectrum_tensor.device,
    )
