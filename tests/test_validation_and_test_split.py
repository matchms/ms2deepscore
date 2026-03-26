import numpy as np
import pytest
from matchms import Spectrum

from ms2deepscore.train_new_model.validation_and_test_split import (
    select_spectra_belonging_to_inchikey,
    select_unique_inchikeys,
    split_spectra_in_random_inchikey_sets,
)


def _inchikey(letter: str) -> str:
    return 14 * letter


def _make_spectrum(letter: str, idx: int = 0) -> Spectrum:
    return Spectrum(
        mz=np.array([100.1 + idx]),
        intensities=np.array([0.9]),
        metadata={"inchikey": _inchikey(letter)},
    )


@pytest.fixture
def sample_spectra():
    return [
        _make_spectrum("A", 0),
        _make_spectrum("B", 1),
        _make_spectrum("B", 2),
        _make_spectrum("C", 3),
    ]


@pytest.fixture
def larger_sample_spectra():
    spectra = []
    # 8 unique inchikeys, 2 spectra each
    for letter in "ABCDEFGH":
        spectra.append(_make_spectrum(letter, 0))
        spectra.append(_make_spectrum(letter, 1))
    return spectra


def _unique_inchikeys_in_spectra(spectra):
    return sorted({s.get("inchikey")[:14] for s in spectra})


def test_select_unique_inchikeys(sample_spectra):
    result = select_unique_inchikeys(sample_spectra)
    assert result == [_inchikey("A"), _inchikey("B"), _inchikey("C")]


def test_select_spectra_belonging_to_inchikey(sample_spectra):
    inchikeys = [_inchikey("A"), _inchikey("B")]
    result = select_spectra_belonging_to_inchikey(sample_spectra, inchikeys)
    assert len(result) == 3
    assert result[0].get("inchikey") == _inchikey("A")
    assert all(s.get("inchikey")[:14] in inchikeys for s in result)


def test_select_spectra_belonging_to_inchikey_empty_match(sample_spectra):
    result = select_spectra_belonging_to_inchikey(sample_spectra, [_inchikey("Z")])
    assert result == []


def test_split_spectra_in_random_inchikey_sets_preserves_all_spectra(sample_spectra):
    val, test, train = split_spectra_in_random_inchikey_sets(sample_spectra, 2, 42)
    assert len(val) + len(test) + len(train) == len(sample_spectra)


def test_split_spectra_in_random_inchikey_sets_splits_by_inchikey_group(larger_sample_spectra):
    val, test, train = split_spectra_in_random_inchikey_sets(larger_sample_spectra, 4, 42)

    val_keys = set(_unique_inchikeys_in_spectra(val))
    test_keys = set(_unique_inchikeys_in_spectra(test))
    train_keys = set(_unique_inchikeys_in_spectra(train))

    assert val_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(train_keys)
    assert test_keys.isdisjoint(train_keys)

    all_keys = val_keys | test_keys | train_keys
    assert all_keys == set(_unique_inchikeys_in_spectra(larger_sample_spectra))


def test_split_spectra_in_random_inchikey_sets_expected_unique_group_sizes(larger_sample_spectra):
    val, test, train = split_spectra_in_random_inchikey_sets(larger_sample_spectra, 4, 42)

    # 8 unique inchikeys, k=4 -> fraction_size = 2
    assert len(_unique_inchikeys_in_spectra(val)) == 2
    assert len(_unique_inchikeys_in_spectra(test)) == 2
    assert len(_unique_inchikeys_in_spectra(train)) == 4

    # two spectra per inchikey
    assert len(val) == 4
    assert len(test) == 4
    assert len(train) == 8


def test_split_spectra_in_random_inchikey_sets_same_seed_is_stable(larger_sample_spectra):
    val1, test1, train1 = split_spectra_in_random_inchikey_sets(larger_sample_spectra, 4, 42)
    val2, test2, train2 = split_spectra_in_random_inchikey_sets(larger_sample_spectra, 4, 42)

    assert _unique_inchikeys_in_spectra(val1) == _unique_inchikeys_in_spectra(val2)
    assert _unique_inchikeys_in_spectra(test1) == _unique_inchikeys_in_spectra(test2)
    assert _unique_inchikeys_in_spectra(train1) == _unique_inchikeys_in_spectra(train2)


def test_split_spectra_in_random_inchikey_sets_different_seed_can_change_split(larger_sample_spectra):
    val1, test1, train1 = split_spectra_in_random_inchikey_sets(larger_sample_spectra, 4, 1)
    val2, test2, train2 = split_spectra_in_random_inchikey_sets(larger_sample_spectra, 4, 2)

    split1 = (
        _unique_inchikeys_in_spectra(val1),
        _unique_inchikeys_in_spectra(test1),
        _unique_inchikeys_in_spectra(train1),
    )
    split2 = (
        _unique_inchikeys_in_spectra(val2),
        _unique_inchikeys_in_spectra(test2),
        _unique_inchikeys_in_spectra(train2),
    )

    assert split1 != split2


def test_split_spectra_in_random_inchikey_sets_none_seed_still_preserves_partition(larger_sample_spectra):
    val, test, train = split_spectra_in_random_inchikey_sets(larger_sample_spectra, 4, None)

    val_keys = set(_unique_inchikeys_in_spectra(val))
    test_keys = set(_unique_inchikeys_in_spectra(test))
    train_keys = set(_unique_inchikeys_in_spectra(train))

    assert val_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(train_keys)
    assert test_keys.isdisjoint(train_keys)
    assert len(val) + len(test) + len(train) == len(larger_sample_spectra)
