import numpy as np
import pytest
from ms2deepscore.vector_operations import (cosine_similarity,
                                            cosine_similarity_matrix,
                                            iqr_pooling, mean_pooling,
                                            median_pooling, std_pooling)


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity(numba_compiled):
    """Test cosine similarity score calculation."""
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])

    if numba_compiled:
        score11 = cosine_similarity(vector1, vector1)
        score12 = cosine_similarity(vector1, vector2)
        score22 = cosine_similarity(vector2, vector2)
    else:
        score11 = cosine_similarity.py_func(vector1, vector1)
        score12 = cosine_similarity.py_func(vector1, vector2)
        score22 = cosine_similarity.py_func(vector2, vector2)

    assert score12 == 2 / np.sqrt(2 * 4), "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_all_zeros(numba_compiled):
    """Test cosine similarity score calculation with empty vector."""
    vector1 = np.array([0, 0, 0, 0])
    vector2 = np.array([1, 1, 1, 1])

    if numba_compiled:
        score11 = cosine_similarity(vector1, vector1)
        score12 = cosine_similarity(vector1, vector2)
        score22 = cosine_similarity(vector2, vector2)
    else:
        score11 = cosine_similarity.py_func(vector1, vector1)
        score12 = cosine_similarity.py_func(vector1, vector2)
        score22 = cosine_similarity.py_func(vector2, vector2)

    assert score11 == score12 == 0.0, "Expected different score."
    assert score22 == 1.0, "Expected different score."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_matrix(numba_compiled):
    """Test cosine similarity scores calculation using int32 input.."""
    vectors1 = np.array([[1, 1, 0, 0],
                         [1, 0, 1, 1]], dtype=np.int32)
    vectors2 = np.array([[0, 1, 1, 0],
                         [0, 0, 1, 1]], dtype=np.int32)

    if numba_compiled:
        scores = cosine_similarity_matrix(vectors1, vectors2)
    else:
        scores = cosine_similarity_matrix.py_func(vectors1, vectors2)
    expected_scores = np.array([[0.5, 0.],
                                   [0.40824829, 0.81649658]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_floats_matrix(numba_compiled):
    """Test cosine similarity scores calculation using float64 input.."""
    vectors1 = np.array([[1, 1, 0, 0],
                         [1, 0, 1, 1]], dtype=np.float64)
    vectors2 = np.array([[0, 1, 1, 0],
                         [0, 0, 1, 1]], dtype=np.float64)

    if numba_compiled:
        scores = cosine_similarity_matrix(vectors1, vectors2)
    else:
        scores = cosine_similarity_matrix.py_func(vectors1, vectors2)
    expected_scores = np.array([[0.5, 0.],
                                [0.40824829, 0.81649658]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_matrix_input_cloned(numba_compiled):
    """Test if score implementation clones the input correctly."""
    vectors1 = np.array([[2, 2, 0, 0],
                         [2, 0, 2, 2]])
    vectors2 = np.array([[0, 2, 2, 0],
                         [0, 0, 2, 2]])

    if numba_compiled:
        cosine_similarity_matrix(vectors1, vectors2)
    else:
        cosine_similarity_matrix.py_func(vectors1, vectors2)

    assert np.all(vectors1 == np.array([[2, 2, 0, 0],
                                              [2, 0, 2, 2]])), "Expected unchanged input."


def test_different_input_vector_lengths():
    """Test if correct error is raised."""
    vector1 = np.array([0, 0, 0, 0])
    vector2 = np.array([1, 1, 1, 1, 1])

    with pytest.raises(AssertionError) as msg:
        _ = cosine_similarity(vector1, vector2)

    expected_message = "Input vector must have same shape."
    assert expected_message == str(msg.value), "Expected particular error message."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_mean_pooling(numba_compiled):
    """Test if scores are pooled correctly."""
    scores = np.arange(0, 16).reshape(4, 4)

    if numba_compiled:
        scores_mean = mean_pooling(scores, 2)
    else:
        scores_mean = mean_pooling.py_func(scores, 2)

    scores_expected = np.array([[ 2.5,  4.5],
                                [10.5, 12.5]])
    assert np.allclose(scores_mean, scores_expected, atol=1e-8), \
        "Expected different pooled mean scores"


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_median_pooling(numba_compiled):
    """Test if scores are pooled correctly."""
    scores = np.arange(0, 16).reshape(4, 4)
    scores[0,0] = 10
    scores[2,2] = 0

    if numba_compiled:
        scores_median = median_pooling(scores, 2)
    else:
        scores_median = median_pooling.py_func(scores, 2)

    scores_expected = np.array([[ 4.5,  4.5],
                                [10.5, 12.5]])
    assert np.allclose(scores_median, scores_expected, atol=1e-8), \
        "Expected different pooled mean scores"


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_std_pooling(numba_compiled):
    """Test if scores are pooled correctly."""
    scores = np.arange(0, 16).reshape(4, 4)

    if numba_compiled:
        scores_std = std_pooling(scores, 2)
    else:
        scores_std = std_pooling.py_func(scores, 2)

    std_expected = np.std([0,1,4,5]) * np.ones((2, 2))
    assert np.allclose(scores_std, std_expected, atol=1e-8), \
        "Expected different pooled standard deviations"


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_iqr_pooling(numba_compiled):
    """Test if scores are pooled correctly."""
    scores = np.arange(0, 16).reshape(4, 4)
    scores[0,0] = 10
    scores[2,2] = 0

    if numba_compiled:
        scores_iqr = iqr_pooling(scores, 2)
    else:
        scores_iqr = iqr_pooling.py_func(scores, 2)

    iqr_expected = np.array([[3. , 3.5],
                             [3.5, 6. ]])
    assert np.allclose(scores_iqr, iqr_expected, atol=1e-8), \
        "Expected different pooled interquantile ranges"
