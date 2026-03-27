import numpy as np
import pytest

from ms2deepscore.fingerprint_similarity_computations import (
    compute_tanimoto_similarity_per_bin,
    compute_tanimoto_similarity_per_bin_between_sets,
)


@pytest.fixture
def simple_binary_fingerprints():
    return np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ], dtype=np.bool_)


@pytest.fixture
def simple_count_fingerprints():
    return np.array([
        [2, 1, 0, 0],
        [1, 0, 2, 0],
        [0, 1, 2, 1],
        [0, 0, 1, 2],
    ], dtype=np.float32)


@pytest.fixture
def simple_sparse_binary_fingerprints():
    return [
        np.array([0, 1], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
    ]


@pytest.fixture
def simple_sparse_count_fingerprints():
    return [
        (np.array([0, 1], dtype=np.int64), np.array([2.0, 1.0], dtype=np.float32)),
        (np.array([0, 2], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32)),
        (np.array([1, 2, 3], dtype=np.int64), np.array([1.0, 2.0, 1.0], dtype=np.float32)),
        (np.array([2, 3], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32)),
    ]


@pytest.fixture
def simple_binary_fingerprints_between_sets():
    fingerprints_1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
    ], dtype=np.bool_)
    fingerprints_2 = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 0],
    ], dtype=np.bool_)
    return fingerprints_1, fingerprints_2


@pytest.fixture
def simple_count_fingerprints_between_sets():
    fingerprints_1 = np.array([
        [2, 0, 0, 0],
        [0, 1, 2, 0],
    ], dtype=np.float32)
    fingerprints_2 = np.array([
        [0, 1, 2, 0],
        [2, 0, 0, 0],
    ], dtype=np.float32)
    return fingerprints_1, fingerprints_2


@pytest.fixture
def simple_sparse_binary_fingerprints_between_sets():
    fingerprints_1 = [
        np.array([0], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
    ]
    fingerprints_2 = [
        np.array([1, 2], dtype=np.int64),
        np.array([0], dtype=np.int64),
    ]
    return fingerprints_1, fingerprints_2


@pytest.fixture
def simple_sparse_count_fingerprints_between_sets():
    fingerprints_1 = [
        (np.array([0], dtype=np.int64), np.array([2.0], dtype=np.float32)),
        (np.array([1, 2], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32)),
    ]
    fingerprints_2 = [
        (np.array([1, 2], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32)),
        (np.array([0], dtype=np.int64), np.array([2.0], dtype=np.float32)),
    ]
    return fingerprints_1, fingerprints_2


def _check_similarity_per_bin_outputs(selected_pairs_per_bin, selected_scores_per_bin, nr_of_items, nr_of_bins, max_pairs_per_bin):
    assert selected_pairs_per_bin.shape == (nr_of_bins, nr_of_items, max_pairs_per_bin)
    assert selected_scores_per_bin.shape == (nr_of_bins, nr_of_items, max_pairs_per_bin)
    assert np.all(selected_scores_per_bin[selected_pairs_per_bin == -1] == 0)
    assert np.all(selected_scores_per_bin >= 0.0)
    assert np.all(selected_scores_per_bin <= 1.0)


def _check_between_sets_similarity_per_bin_outputs(
    selected_pairs_per_bin,
    selected_scores_per_bin,
    nr_of_items_1,
    nr_of_items_2,
    nr_of_bins,
    max_pairs_per_bin,
):
    total_items = nr_of_items_1 + nr_of_items_2
    assert selected_pairs_per_bin.shape == (nr_of_bins, total_items, max_pairs_per_bin)
    assert selected_scores_per_bin.shape == (nr_of_bins, total_items, max_pairs_per_bin)
    assert np.all(selected_scores_per_bin[selected_pairs_per_bin == -1] == 0)
    assert np.all(selected_scores_per_bin >= 0.0)
    assert np.all(selected_scores_per_bin <= 1.0)


@pytest.mark.parametrize(
    "fingerprints_fixture,fingerprint_type",
    [
        ("simple_binary_fingerprints", "rdkit_binary"),
        ("simple_count_fingerprints", "rdkit_count"),
        ("simple_sparse_binary_fingerprints", "rdkit_binary_unfolded"),
        ("simple_sparse_count_fingerprints", "rdkit_count_unfolded"),
    ],
)
def test_compute_tanimoto_similarity_per_bin_all_supported_types(
    request, fingerprints_fixture, fingerprint_type
):
    fingerprints = request.getfixturevalue(fingerprints_fixture)
    max_pairs_per_bin = 5
    nr_of_bins = 10
    selection_bins = np.array(
        [(x / nr_of_bins, x / nr_of_bins + 1 / nr_of_bins) for x in range(nr_of_bins)],
        dtype=np.float32,
    )

    selected_pairs_per_bin, selected_scores_per_bin = compute_tanimoto_similarity_per_bin(
        fingerprints,
        max_pairs_per_bin=max_pairs_per_bin,
        fingerprint_type=fingerprint_type,
        selection_bins=selection_bins,
        include_diagonal=True,
    )

    _check_similarity_per_bin_outputs(
        selected_pairs_per_bin,
        selected_scores_per_bin,
        nr_of_items=len(fingerprints),
        nr_of_bins=nr_of_bins,
        max_pairs_per_bin=max_pairs_per_bin,
    )


@pytest.mark.parametrize(
    "fingerprints_fixture,fingerprint_type",
    [
        ("simple_binary_fingerprints", "rdkit_binary"),
        ("simple_count_fingerprints", "rdkit_count"),
        ("simple_sparse_binary_fingerprints", "rdkit_binary_unfolded"),
        ("simple_sparse_count_fingerprints", "rdkit_count_unfolded"),
    ],
)
def test_compute_tanimoto_similarity_per_bin_exclude_diagonal_all_supported_types(
    request, fingerprints_fixture, fingerprint_type
):
    fingerprints = request.getfixturevalue(fingerprints_fixture)
    max_pairs_per_bin = 5
    nr_of_bins = 10
    selection_bins = np.array(
        [(x / nr_of_bins, x / nr_of_bins + 1 / nr_of_bins) for x in range(nr_of_bins)],
        dtype=np.float32,
    )

    selected_pairs_per_bin, _ = compute_tanimoto_similarity_per_bin(
        fingerprints,
        max_pairs_per_bin=max_pairs_per_bin,
        fingerprint_type=fingerprint_type,
        selection_bins=selection_bins,
        include_diagonal=False,
    )

    for bin_id, pairs_matrix in enumerate(selected_pairs_per_bin):
        for inchikey_idx, row in enumerate(pairs_matrix):
            assert len(np.where(row == inchikey_idx)[0]) == 0, (
                f"Diagonal pair found in bin {bin_id} for item {inchikey_idx}"
            )


def test_compute_tanimoto_similarity_per_bin_binary_dense_expected_pattern(simple_binary_fingerprints):
    max_pairs_per_bin = 5
    nr_of_bins = 10
    selection_bins = np.array(
        [(x / nr_of_bins, x / nr_of_bins + 1 / nr_of_bins) for x in range(nr_of_bins)],
        dtype=np.float32,
    )

    selected_pairs_per_bin, selected_scores_per_bin = compute_tanimoto_similarity_per_bin(
        simple_binary_fingerprints,
        max_pairs_per_bin=max_pairs_per_bin,
        fingerprint_type="rdkit_binary",
        selection_bins=selection_bins,
        include_diagonal=True,
    )

    expected_nr_of_pairs_per_bin = np.array([0, 0, 4, 4, 0, 0, 2, 0, 0, 4])

    for bin_id, pairs_matrix in enumerate(selected_pairs_per_bin):
        number_of_pairs_in_bin = len(np.where(pairs_matrix != -1)[0])
        assert expected_nr_of_pairs_per_bin[bin_id] == number_of_pairs_in_bin

        assert np.all(selected_scores_per_bin[bin_id][pairs_matrix == -1] == 0)
        assert np.all(selected_scores_per_bin[bin_id][pairs_matrix != -1] > 0.0)

        if selection_bins[bin_id][1] == 1:
            for inchikey_idx, row in enumerate(pairs_matrix):
                assert len(np.where(row == inchikey_idx)[0]) == 1
                assert selected_scores_per_bin[bin_id][inchikey_idx][row == inchikey_idx] == 1.0
        else:
            for inchikey_idx, row in enumerate(pairs_matrix):
                assert len(np.where(row == inchikey_idx)[0]) == 0


def test_compute_tanimoto_similarity_per_bin_count_dense_zero_scores_for_invalid_slots(simple_count_fingerprints):
    max_pairs_per_bin = 5
    nr_of_bins = 10
    selection_bins = np.array(
        [(x / nr_of_bins, x / nr_of_bins + 1 / nr_of_bins) for x in range(nr_of_bins)],
        dtype=np.float32,
    )

    selected_pairs_per_bin, selected_scores_per_bin = compute_tanimoto_similarity_per_bin(
        simple_count_fingerprints,
        max_pairs_per_bin=max_pairs_per_bin,
        fingerprint_type="rdkit_count",
        selection_bins=selection_bins,
        include_diagonal=True,
    )

    assert np.all(selected_scores_per_bin[selected_pairs_per_bin == -1] == 0)


@pytest.mark.parametrize(
    "fingerprints_fixture_1,fingerprints_fixture_2,fingerprint_type",
    [
        ("simple_binary_fingerprints_between_sets", "simple_binary_fingerprints_between_sets", "rdkit_binary"),
        ("simple_count_fingerprints_between_sets", "simple_count_fingerprints_between_sets", "rdkit_count"),
        ("simple_sparse_binary_fingerprints_between_sets", "simple_sparse_binary_fingerprints_between_sets", "rdkit_binary_unfolded"),
        ("simple_sparse_count_fingerprints_between_sets", "simple_sparse_count_fingerprints_between_sets", "rdkit_count_unfolded"),
    ],
)
def test_compute_tanimoto_similarity_per_bin_between_sets_all_supported_types(
    request, fingerprints_fixture_1, fingerprints_fixture_2, fingerprint_type
):
    fingerprints_1, fingerprints_2 = request.getfixturevalue(fingerprints_fixture_1)
    max_pairs_per_bin = 2
    selection_bins = np.array([(0.99, 1.0)], dtype=np.float32)

    selected_pairs_per_bin, selected_scores_per_bin = compute_tanimoto_similarity_per_bin_between_sets(
        fingerprints_1,
        fingerprints_2,
        max_pairs_per_bin=max_pairs_per_bin,
        fingerprint_type=fingerprint_type,
        selection_bins=selection_bins,
    )

    _check_between_sets_similarity_per_bin_outputs(
        selected_pairs_per_bin,
        selected_scores_per_bin,
        nr_of_items_1=len(fingerprints_1),
        nr_of_items_2=len(fingerprints_2),
        nr_of_bins=1,
        max_pairs_per_bin=max_pairs_per_bin,
    )


@pytest.mark.parametrize(
    "fingerprints_fixture,fingerprint_type",
    [
        ("simple_binary_fingerprints_between_sets", "rdkit_binary"),
        ("simple_count_fingerprints_between_sets", "rdkit_count"),
        ("simple_sparse_binary_fingerprints_between_sets", "rdkit_binary_unfolded"),
        ("simple_sparse_count_fingerprints_between_sets", "rdkit_count_unfolded"),
    ],
)
def test_compute_tanimoto_similarity_per_bin_between_sets_uses_cross_set_similarity_for_both_directions(
    request, fingerprints_fixture, fingerprint_type
):
    """
    This catches the old bug where the second loop compared set2 against set2
    instead of set2 against set1.
    """
    fingerprints_1, fingerprints_2 = request.getfixturevalue(fingerprints_fixture)

    selection_bins = np.array([(0.99, 1.0)], dtype=np.float32)

    selected_pairs_per_bin, selected_scores_per_bin = compute_tanimoto_similarity_per_bin_between_sets(
        fingerprints_1,
        fingerprints_2,
        max_pairs_per_bin=2,
        fingerprint_type=fingerprint_type,
        selection_bins=selection_bins,
    )

    pairs = selected_pairs_per_bin[0]
    scores = selected_scores_per_bin[0]

    # rows 0..1 belong to set 1
    # rows 2..3 belong to set 2
    #
    # fingerprints are arranged so that:
    # set1[0] <-> set2[1]
    # set1[1] <-> set2[0]

    assert pairs[0, 0] == 3
    assert scores[0, 0] == 1.0

    assert pairs[1, 0] == 2
    assert scores[1, 0] == 1.0

    assert pairs[2, 0] == 1
    assert scores[2, 0] == 1.0

    assert pairs[3, 0] == 0
    assert scores[3, 0] == 1.0

    assert np.all(pairs[:, 1] == -1)
    assert np.all(scores[:, 1] == 0.0)


def test_compute_tanimoto_similarity_per_bin_invalid_fingerprint_type_raises(simple_binary_fingerprints):
    with pytest.raises(ValueError, match="Unsupported fingerprint type"):
        compute_tanimoto_similarity_per_bin(
            simple_binary_fingerprints,
            max_pairs_per_bin=5,
            fingerprint_type="daylight",
            selection_bins=np.array([(-0.01, 1.0)], dtype=np.float32),
            include_diagonal=True,
        )


def test_compute_tanimoto_similarity_per_bin_between_sets_invalid_fingerprint_type_raises(
    simple_binary_fingerprints_between_sets,
):
    fingerprints_1, fingerprints_2 = simple_binary_fingerprints_between_sets
    with pytest.raises(ValueError, match="Unsupported fingerprint type"):
        compute_tanimoto_similarity_per_bin_between_sets(
            fingerprints_1,
            fingerprints_2,
            max_pairs_per_bin=5,
            fingerprint_type="daylight",
            selection_bins=np.array([(-0.01, 1.0)], dtype=np.float32),
        )
