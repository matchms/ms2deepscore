import numpy as np
import pytest
import torch
from torch import equal, tensor

from ms2deepscore.train_new_model.data_augmentation import (
    data_augmentation,
    peak_addition_for_data_augmentation,
    peak_removal_for_data_augmentation,
    change_peak_intensity_for_data_augmentation,
)


# -------------------------------------------------------------------------
# Peak removal
# -------------------------------------------------------------------------


def test_peak_removal_for_data_augmentation():
    spectrum_tensor = tensor(
        [0.0, 0.12, 0.05, 0.78, 0.0, 0.34, 1.0, 0.0, 0.27, 0.65]
    )

    peak_removal_for_data_augmentation(
        spectrum_tensor,
        augment_removal_max=0.5,
        augment_removal_intensity=0.3,
        random_number_generator=np.random.default_rng(123),
    )

    assert equal(
        spectrum_tensor,
        tensor([0.0, 0.12, 0.0, 0.78, 0.0, 0.34, 1.0, 0.0, 0.27, 0.65]),
    )


def test_peak_removal_max_zero_does_not_remove_peaks():
    """A maximum removal fraction of zero should disable peak removal."""
    original = tensor([0.0, 0.05, 0.10, 0.20, 0.50, 1.0])
    augmented = original.clone()

    peak_removal_for_data_augmentation(
        augmented,
        augment_removal_max=0.0,
        augment_removal_intensity=0.3,
        random_number_generator=np.random.default_rng(42),
    )

    assert equal(augmented, original)


def test_peak_removal_only_removes_peaks_below_threshold():
    """Peaks at or above the intensity threshold must never be removed."""
    original = tensor([0.0, 0.05, 0.10, 0.29, 0.30, 0.31, 0.70, 1.0])
    augmented = original.clone()

    peak_removal_for_data_augmentation(
        augmented,
        augment_removal_max=1.0,
        augment_removal_intensity=0.3,
        random_number_generator=np.random.default_rng(12),
    )

    # Originally empty bins stay empty.
    assert augmented[0] == 0.0

    # Threshold is exclusive: 0.30 itself must not be removed.
    assert augmented[4] == original[4]

    # Peaks above the threshold must remain unchanged.
    assert equal(augmented[4:], original[4:])


def test_peak_removal_respects_maximum_fraction():
    """Never remove more than augment_removal_max of eligible peaks."""
    nr_peaks = 100
    augment_removal_max = 0.2

    spectrum_tensor = torch.full((nr_peaks,), 0.1)

    peak_removal_for_data_augmentation(
        spectrum_tensor,
        augment_removal_max=augment_removal_max,
        augment_removal_intensity=0.3,
        random_number_generator=np.random.default_rng(42),
    )

    nr_removed = int((spectrum_tensor == 0).sum())

    assert nr_removed <= nr_peaks * augment_removal_max


@pytest.mark.parametrize(
    "augment_removal_intensity",
    [0.0, -0.1],
)
def test_peak_removal_with_no_eligible_peaks_changes_nothing(
    augment_removal_intensity,
):
    original = tensor([0.0, 0.1, 0.2, 0.5, 1.0])
    augmented = original.clone()

    peak_removal_for_data_augmentation(
        augmented,
        augment_removal_max=0.5,
        augment_removal_intensity=augment_removal_intensity,
        random_number_generator=np.random.default_rng(42),
    )

    assert equal(augmented, original)


# -------------------------------------------------------------------------
# Peak addition
# -------------------------------------------------------------------------


def test_peak_addition_for_data_augmentation():
    spectrum_tensor = tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.27, 0.0]
    )

    peak_addition_for_data_augmentation(
        spectrum_tensor,
        4,
        0.02,
        random_number_generator=np.random.default_rng(0),
    )

    assert spectrum_tensor[6] == 1.0
    assert spectrum_tensor[8] == 0.27
    assert spectrum_tensor[0] == 0.0
    assert spectrum_tensor[2] != 0.0


def test_peak_addition_does_not_modify_existing_peaks():
    original = tensor([0.0, 0.2, 0.0, 0.7, 0.0, 1.0, 0.0])
    augmented = original.clone()

    peak_addition_for_data_augmentation(
        augmented,
        augment_noise_max=5,
        augment_noise_intensity=0.05,
        random_number_generator=np.random.default_rng(4),
    )

    existing_peak_mask = original > 0

    assert equal(
        augmented[existing_peak_mask],
        original[existing_peak_mask],
    )


def test_peak_addition_only_adds_peaks_to_empty_bins():
    original = tensor([0.0, 0.2, 0.0, 0.7, 0.0, 1.0, 0.0])
    augmented = original.clone()

    peak_addition_for_data_augmentation(
        augmented,
        augment_noise_max=5,
        augment_noise_intensity=0.05,
        random_number_generator=np.random.default_rng(8),
    )

    changed = augmented != original

    # Every changed bin must have been zero before augmentation.
    assert torch.all(original[changed] == 0)


def test_peak_addition_respects_maximum_number_of_new_peaks():
    spectrum_tensor = torch.zeros(100)

    augment_noise_max = 10

    peak_addition_for_data_augmentation(
        spectrum_tensor,
        augment_noise_max=augment_noise_max,
        augment_noise_intensity=0.05,
        random_number_generator=np.random.default_rng(42),
    )

    nr_added = int((spectrum_tensor > 0).sum())

    assert nr_added <= augment_noise_max


def test_peak_addition_respects_maximum_noise_intensity():
    spectrum_tensor = torch.zeros(100)

    augment_noise_intensity = 0.05

    peak_addition_for_data_augmentation(
        spectrum_tensor,
        augment_noise_max=20,
        augment_noise_intensity=augment_noise_intensity,
        random_number_generator=np.random.default_rng(42),
    )

    assert torch.all(spectrum_tensor >= 0)
    assert torch.all(spectrum_tensor <= augment_noise_intensity)


def test_peak_addition_max_zero_changes_nothing():
    original = tensor([0.0, 0.2, 0.0, 0.7, 0.0])
    augmented = original.clone()

    peak_addition_for_data_augmentation(
        augmented,
        augment_noise_max=0,
        augment_noise_intensity=0.05,
        random_number_generator=np.random.default_rng(42),
    )

    assert equal(augmented, original)


def test_peak_addition_when_no_empty_bins_changes_nothing():
    original = tensor([0.1, 0.2, 0.5, 0.7, 1.0])
    augmented = original.clone()

    peak_addition_for_data_augmentation(
        augmented,
        augment_noise_max=10,
        augment_noise_intensity=0.05,
        random_number_generator=np.random.default_rng(42),
    )

    assert equal(augmented, original)


# -------------------------------------------------------------------------
# Peak intensity changes
# -------------------------------------------------------------------------


def test_change_peak_intensity_for_data_augmentation():
    spectrum_tensor = tensor(
        [0.0, 0.12, 0.05, 0.78, 0.0, 0.34, 1.0, 0.0, 0.27, 0.65]
    )

    change_peak_intensity_for_data_augmentation(
        spectrum_tensor,
        0.2,
    )

    assert spectrum_tensor[0] == 0.0
    assert spectrum_tensor[1] != 0.12


def test_change_peak_intensity_zero_changes_nothing():
    original = tensor([0.0, 0.12, 0.4, 1.0])
    augmented = original.clone()

    change_peak_intensity_for_data_augmentation(
        augmented,
        augment_intensity=0.0,
    )

    assert equal(augmented, original)


def test_change_peak_intensity_preserves_zero_bins():
    spectrum_tensor = tensor([0.0, 0.1, 0.0, 0.4, 0.0, 1.0])

    change_peak_intensity_for_data_augmentation(
        spectrum_tensor,
        augment_intensity=0.2,
    )

    assert equal(
        spectrum_tensor[[0, 2, 4]],
        tensor([0.0, 0.0, 0.0]),
    )


def test_change_peak_intensity_stays_within_requested_range():
    """A=0.2 means intensities may change by at most +/-20%."""
    original = tensor([0.1, 0.2, 0.5, 0.8, 1.0])
    augmented = original.clone()

    augment_intensity = 0.2

    change_peak_intensity_for_data_augmentation(
        augmented,
        augment_intensity=augment_intensity,
    )

    lower_bound = original * (1 - augment_intensity)
    upper_bound = original * (1 + augment_intensity)

    assert torch.all(augmented >= lower_bound)
    assert torch.all(augmented <= upper_bound)


def test_change_peak_intensity_does_not_create_or_remove_peaks():
    original = tensor([0.0, 0.1, 0.0, 0.5, 1.0, 0.0])
    augmented = original.clone()

    change_peak_intensity_for_data_augmentation(
        augmented,
        augment_intensity=0.2,
    )

    assert equal(augmented == 0, original == 0)


# -------------------------------------------------------------------------
# Complete augmentation pipeline
# -------------------------------------------------------------------------


def test_data_augmentation_preserves_shape_and_dtype():
    spectra = tensor(
        [
            [0.0, 0.1, 0.2, 0.0, 1.0],
            [0.3, 0.0, 0.2, 0.5, 0.0],
        ]
    )

    original_shape = spectra.shape
    original_dtype = spectra.dtype

    settings = _DummyAugmentationSettings()

    data_augmentation(
        spectra,
        settings,
        np.random.default_rng(42),
    )

    assert spectra.shape == original_shape
    assert spectra.dtype == original_dtype


def test_data_augmentation_is_reproducible_when_both_rngs_are_seeded():
    """The implementation uses both NumPy and PyTorch random generators."""
    original = tensor(
        [
            [0.0, 0.05, 0.2, 0.0, 1.0],
            [0.1, 0.0, 0.25, 0.5, 0.0],
        ]
    )

    settings = _DummyAugmentationSettings()

    first = original.clone()
    torch.manual_seed(123)
    data_augmentation(
        first,
        settings,
        np.random.default_rng(456),
    )

    second = original.clone()
    torch.manual_seed(123)
    data_augmentation(
        second,
        settings,
        np.random.default_rng(456),
    )

    assert equal(first, second)


class _DummyAugmentationSettings:
    augment_removal_max = 0.2
    augment_removal_intensity = 0.3
    augment_intensity = 0.2
    augment_noise_max = 4
    augment_noise_intensity = 0.02