import numpy as np
import torch
from matchms import Spectrum

from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.data_augmentation import (data_augmentation, data_augmentation_spectrum,
                                                            peak_addition_for_data_augmentation,
                                                            peak_removal_for_data_augmentation, change_peak_intensity_for_data_augmentation)

def test_peak_removal_for_data_augmentation():
    spectrum_tensor = torch.tensor([0.0, 0.12, 0.05, 0.78, 0.0, 0.34, 1.0, 0.0, 0.27, 0.65])
    peak_removal_for_data_augmentation(spectrum_tensor,
                                       augment_removal_max=0.5 ,
                                       augment_removal_intensity=0.3,
                                       random_number_generator= np.random.default_rng(42))
    assert torch.equal(spectrum_tensor, torch.tensor([0.0, 0.12, 0.0, 0.78, 0.0, 0.34, 1.0, 0.0, 0.0, 0.65]))

def test_peak_addition_for_data_augmentation():
    spectrum_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.27, 0.0])
    peak_addition_for_data_augmentation(spectrum_tensor,
                                       4,
                                       0.02,
                                       random_number_generator= np.random.default_rng(0))
    assert spectrum_tensor[6] == 1.0
    assert spectrum_tensor[8] == 0.27
    assert spectrum_tensor[0] == 0.0
    assert spectrum_tensor[2] != 0.0 # we know this one is changed because of the random number generator

def test_change_peak_intensity_for_data_augmentation():
    spectrum_tensor = torch.tensor([0.0, 0.12, 0.05, 0.78, 0.0, 0.34, 1.0, 0.0, 0.27, 0.65])
    change_peak_intensity_for_data_augmentation(spectrum_tensor,
                                       0.2)
    assert spectrum_tensor[0] == 0.0 # Check that zero's are not changed.
    assert spectrum_tensor[1] != 0.12 # Check that the value is changed.
