import numpy as np
import pytest

from ms2deepscore.fingerprint_utils import (
    _inchi_to_smiles,
    normalize_to_smiles,
    derive_fingerprint_from_smiles,
    derive_fingerprint_from_smiles_or_inchi,
)


VALID_SMILES = "CCO"
VALID_SMILES_2 = "CCCO"
VALID_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
INVALID_INCHI = "InChI=1S/this_is_not_valid"


def _assert_unfolded_binary_single(fp):
    assert isinstance(fp, np.ndarray)
    assert fp.ndim == 1
    assert fp.shape[0] > 0
    assert np.issubdtype(fp.dtype, np.integer)


def _assert_unfolded_binary_list(fps, expected_len):
    assert isinstance(fps, list)
    assert len(fps) == expected_len
    for fp in fps:
        _assert_unfolded_binary_single(fp)


def _assert_unfolded_count_single(fp):
    assert isinstance(fp, tuple)
    assert len(fp) == 2

    bins, counts = fp
    assert isinstance(bins, np.ndarray)
    assert isinstance(counts, np.ndarray)
    assert bins.ndim == 1
    assert counts.ndim == 1
    assert bins.shape == counts.shape
    assert bins.shape[0] > 0
    assert np.issubdtype(bins.dtype, np.integer)
    assert np.all(counts > 0)


def _assert_unfolded_count_list(fps, expected_len):
    assert isinstance(fps, list)
    assert len(fps) == expected_len
    for fp in fps:
        _assert_unfolded_count_single(fp)


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
def test_derive_fingerprint_from_smiles_single_folded(fingerprint_type):
    fp = derive_fingerprint_from_smiles(
        VALID_SMILES,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (256,)
    assert np.sum(fp) > 0


@pytest.mark.parametrize("fingerprint_type", ["rdkit_binary", "rdkit_count"])
def test_derive_fingerprint_from_smiles_list_folded(fingerprint_type):
    fps = derive_fingerprint_from_smiles(
        [VALID_SMILES, VALID_SMILES_2],
        fingerprint_type=fingerprint_type,
        nbits=256,
    )
    assert isinstance(fps, np.ndarray)
    assert fps.shape == (2, 256)
    assert np.all(fps.sum(axis=1) > 0)


def test_derive_fingerprint_from_smiles_single_binary_unfolded():
    fp = derive_fingerprint_from_smiles(
        VALID_SMILES,
        fingerprint_type="rdkit_binary_unfolded",
        nbits=256,
    )
    _assert_unfolded_binary_single(fp)


def test_derive_fingerprint_from_smiles_list_binary_unfolded():
    fps = derive_fingerprint_from_smiles(
        [VALID_SMILES, VALID_SMILES_2],
        fingerprint_type="rdkit_binary_unfolded",
        nbits=256,
    )
    _assert_unfolded_binary_list(fps, expected_len=2)


def test_derive_fingerprint_from_smiles_single_count_unfolded():
    fp = derive_fingerprint_from_smiles(
        VALID_SMILES,
        fingerprint_type="rdkit_count_unfolded",
        nbits=256,
    )
    _assert_unfolded_count_single(fp)


def test_derive_fingerprint_from_smiles_list_count_unfolded():
    fps = derive_fingerprint_from_smiles(
        [VALID_SMILES, VALID_SMILES_2],
        fingerprint_type="rdkit_count_unfolded",
        nbits=256,
    )
    _assert_unfolded_count_list(fps, expected_len=2)


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


@pytest.mark.parametrize("fingerprint_type", ["rdkit_binary", "rdkit_count"])
def test_derive_fingerprint_from_smiles_or_inchi_single_smiles_folded(fingerprint_type):
    fp = derive_fingerprint_from_smiles_or_inchi(
        VALID_SMILES,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (256,)
    assert np.sum(fp) > 0


@pytest.mark.parametrize("fingerprint_type", ["rdkit_binary", "rdkit_count"])
def test_derive_fingerprint_from_smiles_or_inchi_single_inchi_folded(fingerprint_type):
    fp = derive_fingerprint_from_smiles_or_inchi(
        VALID_INCHI,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (256,)
    assert np.sum(fp) > 0


def test_derive_fingerprint_from_smiles_or_inchi_single_smiles_binary_unfolded():
    fp = derive_fingerprint_from_smiles_or_inchi(
        VALID_SMILES,
        fingerprint_type="rdkit_binary_unfolded",
        nbits=256,
    )
    _assert_unfolded_binary_single(fp)


def test_derive_fingerprint_from_smiles_or_inchi_single_inchi_binary_unfolded():
    fp = derive_fingerprint_from_smiles_or_inchi(
        VALID_INCHI,
        fingerprint_type="rdkit_binary_unfolded",
        nbits=256,
    )
    _assert_unfolded_binary_single(fp)


def test_derive_fingerprint_from_smiles_or_inchi_single_smiles_count_unfolded():
    fp = derive_fingerprint_from_smiles_or_inchi(
        VALID_SMILES,
        fingerprint_type="rdkit_count_unfolded",
        nbits=256,
    )
    _assert_unfolded_count_single(fp)


def test_derive_fingerprint_from_smiles_or_inchi_single_inchi_count_unfolded():
    fp = derive_fingerprint_from_smiles_or_inchi(
        VALID_INCHI,
        fingerprint_type="rdkit_count_unfolded",
        nbits=256,
    )
    _assert_unfolded_count_single(fp)


@pytest.mark.parametrize("fingerprint_type",
                         ["rdkit_binary", "rdkit_count", "rdkit_logcount"])
def test_derive_fingerprint_from_smiles_or_inchi_list_mixed_valid_folded(fingerprint_type):
    fps = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, VALID_INCHI, VALID_SMILES_2],
        fingerprint_type=fingerprint_type,
        nbits=256,
    )
    assert isinstance(fps, np.ndarray)
    assert fps.shape == (3, 256)
    assert np.all(fps.sum(axis=1) > 0)


def test_derive_fingerprint_count_vs_logcount():
    # test if indeed logcount is approximately log1p of count
    fps_count = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, VALID_INCHI, VALID_SMILES_2],
        fingerprint_type="rdkit_count",
        nbits=256,
    )
    fps_logcount = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, VALID_INCHI, VALID_SMILES_2],
        fingerprint_type="rdkit_logcount",
        nbits=256,
    )
    assert fps_count.shape == fps_logcount.shape
    assert np.log1p(fps_count) == pytest.approx(fps_logcount, rel=1e-5)


def test_derive_fingerprint_from_smiles_or_inchi_list_mixed_valid_binary_unfolded():
    fps = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, VALID_INCHI, VALID_SMILES_2],
        fingerprint_type="rdkit_binary_unfolded",
        nbits=256,
    )
    _assert_unfolded_binary_list(fps, expected_len=3)


@pytest.mark.parametrize("fp_type",
                        ["rdkit_count_unfolded", "rdkit_logcount_unfolded"])
def test_derive_fingerprint_count_and_logcount_unfolded(fp_type):
    fps = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, VALID_INCHI, VALID_SMILES_2],
        fingerprint_type=fp_type,
        nbits=256,
    )
    _assert_unfolded_count_list(fps, expected_len=3)


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
        policy_invalid="keep",
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
        policy_invalid="keep",
    )
    assert isinstance(fps, np.ndarray)
    assert fps.shape == (0, 256)


def test_derive_fingerprint_from_smiles_or_inchi_mixed_list_ignore_drops_invalid_entries_folded():
    fps = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, INVALID_INCHI, VALID_SMILES_2],
        fingerprint_type="rdkit_binary",
        nbits=256,
        policy_invalid="keep",
    )
    assert isinstance(fps, np.ndarray)
    assert fps.shape == (2, 256)
    assert np.all(fps.sum(axis=1) > 0)


def test_derive_fingerprint_from_smiles_or_inchi_mixed_list_ignore_drops_invalid_entries_binary_unfolded():
    fps = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, INVALID_INCHI, VALID_SMILES_2],
        fingerprint_type="rdkit_binary_unfolded",
        nbits=256,
        policy_invalid="keep",
    )
    _assert_unfolded_binary_list(fps, expected_len=2)


def test_derive_fingerprint_from_smiles_or_inchi_mixed_list_ignore_drops_invalid_entries_count_unfolded():
    fps = derive_fingerprint_from_smiles_or_inchi(
        [VALID_SMILES, INVALID_INCHI, VALID_SMILES_2],
        fingerprint_type="rdkit_count_unfolded",
        nbits=256,
        policy_invalid="keep",
    )
    _assert_unfolded_count_list(fps, expected_len=2)
