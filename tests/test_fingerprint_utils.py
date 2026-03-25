import numpy as np
import pytest

from ms2deepscore.fingerprint_utils import (
    _inchi_to_smiles,
    normalize_to_smiles,
    derive_fingerprint_from_smiles,
    derive_fingerprint_from_smiles_or_inchi,
)


VALID_SMILES = "CCO"
VALID_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
INVALID_INCHI = "InChI=1S/this_is_not_valid"


def test_inchi_to_smiles_valid():
    smiles = _inchi_to_smiles(VALID_INCHI)
    assert isinstance(smiles, str)
    assert len(smiles) > 0


def test_inchi_to_smiles_invalid_returns_none():
    smiles = _inchi_to_smiles(INVALID_INCHI)
    assert smiles is None


def test_normalize_to_smiles_single_smiles():
    result = normalize_to_smiles(VALID_SMILES)
    assert result == VALID_SMILES


def test_normalize_to_smiles_single_inchi():
    result = normalize_to_smiles(VALID_INCHI)
    assert isinstance(result, str)
    assert len(result) > 0


def test_normalize_to_smiles_single_invalid_inchi():
    result = normalize_to_smiles(INVALID_INCHI)
    assert result is None


def test_normalize_to_smiles_list_mixed_inputs():
    result = normalize_to_smiles([VALID_SMILES, VALID_INCHI, None, INVALID_INCHI])
    assert isinstance(result, list)
    assert result[0] == VALID_SMILES
    assert isinstance(result[1], str)
    assert result[2] is None
    assert result[3] is None


@pytest.mark.parametrize("fingerprint_type", ["rdkit_binary", "rdkit_count"])
def test_derive_fingerprint_from_smiles_single(fingerprint_type):
    fp = derive_fingerprint_from_smiles(
        VALID_SMILES,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (256,)
    assert np.sum(fp) > 0


@pytest.mark.parametrize("fingerprint_type", ["rdkit_binary", "rdkit_count"])
def test_derive_fingerprint_from_smiles_list(fingerprint_type):
    fps = derive_fingerprint_from_smiles(
        [VALID_SMILES, "CCCO"],
        fingerprint_type=fingerprint_type,
        nbits=256,
    )
    assert isinstance(fps, np.ndarray)
    assert fps.shape == (2, 256)
    assert np.all(fps.sum(axis=1) > 0)


def test_derive_fingerprint_from_smiles_binary_is_integer_like():
    fp = derive_fingerprint_from_smiles(
        VALID_SMILES,
        fingerprint_type="rdkit_binary",
        nbits=256,
    )
    unique_values = np.unique(fp)
    assert set(unique_values).issubset({0, 1})


def test_derive_fingerprint_from_smiles_count_has_correct_shape():
    fp = derive_fingerprint_from_smiles(
        VALID_SMILES,
        fingerprint_type="rdkit_count",
        nbits=256,
    )
    assert fp.shape == (256,)
    assert np.sum(fp) > 0


def test_derive_fingerprint_from_smiles_invalid_fingerprint_type_raises():
    with pytest.raises(ValueError, match="Unsupported fingerprint type"):
        derive_fingerprint_from_smiles(
            VALID_SMILES,
            fingerprint_type="daylight",
            nbits=256,
        )


def test_derive_fingerprint_from_smiles_invalid_smiles_raises():
    with pytest.raises(Exception):
        derive_fingerprint_from_smiles(
            "this_is_not_a_smiles",
            fingerprint_type="rdkit_binary",
            nbits=256,
            policy_invalid_smiles="raise",
        )


def test_derive_fingerprint_from_smiles_or_inchi_single_smiles():
    fp = derive_fingerprint_from_smiles_or_inchi(
        VALID_SMILES,
        fingerprint_type="rdkit_binary",
        nbits=256,
    )
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (256,)
    assert np.sum(fp) > 0


def test_derive_fingerprint_from_smiles_or_inchi_single_inchi():
    fp = derive_fingerprint_from_smiles_or_inchi(
        VALID_INCHI,
        fingerprint_type="rdkit_binary",
        nbits=256,
    )
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (256,)
    assert np.sum(fp) > 0


def test_derive_fingerprint_from_smiles_or_inchi_list_mixed_valid():
    fps = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, VALID_INCHI, "CCCO"],
        fingerprint_type="rdkit_binary",
        nbits=256,
    )
    assert isinstance(fps, np.ndarray)
    assert fps.shape == (3, 256)
    assert np.all(fps.sum(axis=1) > 0)


def test_derive_fingerprint_from_smiles_or_inchi_invalid_single_raises():
    with pytest.raises(ValueError, match="Could not convert input structure to SMILES"):
        derive_fingerprint_from_smiles_or_inchi(
            INVALID_INCHI,
            fingerprint_type="rdkit_binary",
            nbits=256,
            policy_invalid="raise",
        )


def test_derive_fingerprint_from_smiles_or_inchi_invalid_single_ignore_returns_zero_vector():
    fp = derive_fingerprint_from_smiles_or_inchi(
        INVALID_INCHI,
        fingerprint_type="rdkit_binary",
        nbits=256,
        policy_invalid="ignore",
    )
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (256,)
    assert np.all(fp == 0)


def test_derive_fingerprint_from_smiles_or_inchi_all_invalid_list_raises():
    with pytest.raises(ValueError, match="No valid SMILES/InChI entries available for fingerprinting"):
        derive_fingerprint_from_smiles_or_inchi(
            [INVALID_INCHI, INVALID_INCHI],
            fingerprint_type="rdkit_binary",
            nbits=256,
            policy_invalid="raise",
        )


def test_derive_fingerprint_from_smiles_or_inchi_all_invalid_list_ignore_returns_empty_array():
    fps = derive_fingerprint_from_smiles_or_inchi(
        [INVALID_INCHI, INVALID_INCHI],
        fingerprint_type="rdkit_binary",
        nbits=256,
        policy_invalid="ignore",
    )
    assert isinstance(fps, np.ndarray)
    assert fps.shape == (0, 256)


def test_derive_fingerprint_from_smiles_or_inchi_mixed_list_ignore_drops_invalid_entries():
    fps = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, INVALID_INCHI, "CCCO"],
        fingerprint_type="rdkit_binary",
        nbits=256,
        policy_invalid="ignore",
    )
    assert isinstance(fps, np.ndarray)
    assert fps.shape == (2, 256)
    assert np.all(fps.sum(axis=1) > 0)
