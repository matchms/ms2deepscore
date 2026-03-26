import string
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum

from ms2deepscore.validation_loss_calculation.calculate_scores_for_validation import (
    calculate_tanimoto_scores_unique_inchikey,
)

TEST_RESOURCES_PATH = Path(__file__).parent / "resources"


def create_dummy_data(nr_of_spectra):
    """Create fake spectra with valid, distinct SMILES."""
    if nr_of_spectra > 10:
        raise ValueError("This helper currently supports up to 10 spectra.")

    smiles_list = [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCCO",
        "CCCCO",
        "CC(C)O",
        "CC(C)C",
        "CC(C)(C)O",
    ]

    spectrums = []
    for i in range(nr_of_spectra):
        dummy_inchikey = f"{14 * string.ascii_uppercase[i]}-{10 * string.ascii_uppercase[i]}-N"
        spectrum = Spectrum(
            mz=np.array([100.0 + (i + 1) * 25.0]),
            intensities=np.array([0.1]),
            metadata={
                "inchikey": dummy_inchikey,
                "smiles": smiles_list[i],
            },
        )
        spectrums.append(spectrum)
    return spectrums


@pytest.mark.parametrize(
    "fingerprint_type",
    [
        "rdkit_binary",
        "rdkit_count",
        "rdkit_binary_unfolded",
        "rdkit_count_unfolded",
    ],
)
def test_calculate_tanimoto_scores_unique_inchikey_shape(fingerprint_type):
    """Scores should be calculated only between unique InChIKeys."""
    nr_of_test_spectra = 4
    spectrums = create_dummy_data(nr_of_test_spectra)

    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
        spectrums + spectrums,
        spectrums,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )

    assert isinstance(tanimoto_scores, pd.DataFrame)
    assert tanimoto_scores.shape == (nr_of_test_spectra, nr_of_test_spectra)


@pytest.mark.parametrize(
    "fingerprint_type",
    [
        "rdkit_binary",
        "rdkit_count",
        "rdkit_binary_unfolded",
        "rdkit_count_unfolded",
    ],
)
def test_calculate_tanimoto_scores_unique_inchikey_diagonal_is_one(fingerprint_type):
    """Self-similarity should be 1.0 when comparing the same unique structures."""
    spectrums = create_dummy_data(5)

    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
        spectrums,
        spectrums,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )

    diag = np.diag(tanimoto_scores.values)
    assert np.allclose(diag, 1.0)


@pytest.mark.parametrize(
    "fingerprint_type",
    [
        "rdkit_binary",
        "rdkit_count",
        "rdkit_binary_unfolded",
        "rdkit_count_unfolded",
    ],
)
def test_calculate_tanimoto_scores_unique_inchikey_index_and_columns(fingerprint_type):
    """The DataFrame should use 14-character InChIKeys as index/columns."""
    spectrums = create_dummy_data(4)

    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
        spectrums,
        spectrums,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )

    expected_inchikeys = [s.get("inchikey")[:14] for s in spectrums]
    assert list(tanimoto_scores.index) == expected_inchikeys
    assert list(tanimoto_scores.columns) == expected_inchikeys


@pytest.mark.parametrize(
    "fingerprint_type",
    [
        "rdkit_binary",
        "rdkit_count",
        "rdkit_binary_unfolded",
        "rdkit_count_unfolded",
    ],
)
def test_calculate_tanimoto_scores_unique_inchikey_values_in_valid_range(fingerprint_type):
    """All Tanimoto similarities should lie in [0, 1]."""
    spectrums_1 = create_dummy_data(4)
    spectrums_2 = create_dummy_data(3)

    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
        spectrums_1,
        spectrums_2,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )

    assert np.all(tanimoto_scores.values >= 0.0)
    assert np.all(tanimoto_scores.values <= 1.0)


@pytest.mark.parametrize(
    "fingerprint_type",
    [
        "rdkit_binary",
        "rdkit_count",
        "rdkit_binary_unfolded",
        "rdkit_count_unfolded",
    ],
)
def test_calculate_tanimoto_scores_unique_inchikey_nonsymmetric_shape(fingerprint_type):
    """The function should support non-symmetric comparisons."""
    spectrums_1 = create_dummy_data(5)
    spectrums_2 = create_dummy_data(3)

    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
        spectrums_1,
        spectrums_2,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )

    assert tanimoto_scores.shape == (5, 3)


@pytest.mark.parametrize(
    "fingerprint_type",
    [
        "rdkit_binary",
        "rdkit_count",
        "rdkit_binary_unfolded",
        "rdkit_count_unfolded",
    ],
)
def test_calculate_tanimoto_scores_unique_inchikey_duplicate_input_collapses_to_unique(fingerprint_type):
    """Duplicated spectra with the same InChIKey should collapse to one row/column."""
    spectrums = create_dummy_data(4)

    duplicated_input = [spectrums[0], spectrums[0], spectrums[1], spectrums[1], spectrums[2], spectrums[3]]

    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
        duplicated_input,
        spectrums,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )

    assert tanimoto_scores.shape == (4, 4)
    expected_inchikeys = [s.get("inchikey")[:14] for s in spectrums]
    assert list(tanimoto_scores.index) == expected_inchikeys
    assert list(tanimoto_scores.columns) == expected_inchikeys


def test_calculate_tanimoto_scores_unique_inchikey_empty_input_raises():
    spectrums = create_dummy_data(3)

    with pytest.raises(ValueError, match="larger than 0"):
        calculate_tanimoto_scores_unique_inchikey(
            [],
            spectrums,
            fingerprint_type="rdkit_binary",
            nbits=256,
        )

    with pytest.raises(ValueError, match="larger than 0"):
        calculate_tanimoto_scores_unique_inchikey(
            spectrums,
            [],
            fingerprint_type="rdkit_binary",
            nbits=256,
        )
        