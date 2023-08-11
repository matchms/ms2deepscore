import pytest
import numpy as np
from ms2deepscore.spectrum_pair_selection import (
    jaccard_similarity_matrix_cherrypicking,
    compute_jaccard_similarity_matrix_cherrypicking,
)


@pytest.fixture
def simple_fingerprints():
    return np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)


@pytest.fixture
def fingerprints(): 
    return np.array([
        [1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
    ], dtype=bool)


def test_basic_functionality(simple_fingerprints):
    matrix = jaccard_similarity_matrix_cherrypicking(simple_fingerprints, random_seed=42)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix.diagonal(), 1.0)
    assert matrix.nnz > 0  # Make sure there are some non-zero entries


def test_exclude_diagonal(simple_fingerprints):
    matrix = jaccard_similarity_matrix_cherrypicking(simple_fingerprints, include_diagonal=False, random_seed=42)
    diagonal = matrix.diagonal()
    assert np.all(diagonal == 0)  # Ensure no non-zero diagonal elements


def test_correct_counts(fingerprints):
    matrix = jaccard_similarity_matrix_cherrypicking(fingerprints)
    expected_histogram = np.array([6,  8,  2, 10,  8, 14,  0,  8,  0,  8])
    assert np.all(np.histogram(matrix.todense(), 10)[0] == expected_histogram)


def test_global_bias(fingerprints):
    bins = np.array([(0, 0.5), (0.5, 0.8), (0.8, 1.0)])
    data, _, _ = compute_jaccard_similarity_matrix_cherrypicking(fingerprints,
                                                                 selections_bins=bins,
                                                                 max_pairs_per_bin=1)
    data = np.array(data)
    assert (data <= 0.5).sum() == ((data>0.5) & (data<=0.8)).sum() == (data>0.8).sum() == 8


def test_global_bias_not_possible(fingerprints):
    bins = np.array([(0, 0.5), (0.5, 0.8), (0.8, 1.0)])
    data, _, _ = compute_jaccard_similarity_matrix_cherrypicking(fingerprints,
                                                                 selections_bins=bins,
                                                                 max_pairs_per_bin=2)
    data = np.array(data)
    assert (data <= 0.5).sum() == ((data>0.5) & (data<=0.8)).sum() == 16
    assert (data>0.8).sum() == 8
