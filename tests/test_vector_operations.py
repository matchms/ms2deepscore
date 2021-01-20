import os
import numpy
import pytest
from matchms import Spectrum
from ms2deepscore.vector_operations import cosine_similarity
from ms2deepscore.vector_operations import cosine_similarity_matrix


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity(numba_compiled):
    """Test cosine similarity score calculation."""
    vector1 = numpy.array([1, 1, 0, 0])
    vector2 = numpy.array([1, 1, 1, 1])

    if numba_compiled:
        score11 = cosine_similarity(vector1, vector1)
        score12 = cosine_similarity(vector1, vector2)
        score22 = cosine_similarity(vector2, vector2)
    else:
        score11 = cosine_similarity.py_func(vector1, vector1)
        score12 = cosine_similarity.py_func(vector1, vector2)
        score22 = cosine_similarity.py_func(vector2, vector2)

    assert score12 == 2 / numpy.sqrt(2 * 4), "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_all_zeros(numba_compiled):
    """Test cosine similarity score calculation with empty vector."""
    vector1 = numpy.array([0, 0, 0, 0])
    vector2 = numpy.array([1, 1, 1, 1])

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
    vectors1 = numpy.array([[1, 1, 0, 0],
                            [1, 0, 1, 1]], dtype=numpy.int32)
    vectors2 = numpy.array([[0, 1, 1, 0],
                            [0, 0, 1, 1]], dtype=numpy.int32)

    if numba_compiled:
        scores = cosine_similarity_matrix(vectors1, vectors2)
    else:
        scores = cosine_similarity_matrix.py_func(vectors1, vectors2)
    expected_scores = numpy.array([[0.5, 0.],
                                   [0.40824829, 0.81649658]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_floats_matrix(numba_compiled):
    """Test cosine similarity scores calculation using float64 input.."""
    vectors1 = numpy.array([[1, 1, 0, 0],
                            [1, 0, 1, 1]], dtype=numpy.float64)
    vectors2 = numpy.array([[0, 1, 1, 0],
                            [0, 0, 1, 1]], dtype=numpy.float64)

    if numba_compiled:
        scores = cosine_similarity_matrix(vectors1, vectors2)
    else:
        scores = cosine_similarity_matrix.py_func(vectors1, vectors2)
    expected_scores = numpy.array([[0.5, 0.],
                                   [0.40824829, 0.81649658]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_matrix_input_cloned(numba_compiled):
    """Test if score implementation clones the input correctly."""
    vectors1 = numpy.array([[2, 2, 0, 0],
                            [2, 0, 2, 2]])
    vectors2 = numpy.array([[0, 2, 2, 0],
                            [0, 0, 2, 2]])

    if numba_compiled:
        cosine_similarity_matrix(vectors1, vectors2)
    else:
        cosine_similarity_matrix.py_func(vectors1, vectors2)

    assert numpy.all(vectors1 == numpy.array([[2, 2, 0, 0],
                                              [2, 0, 2, 2]])), "Expected unchanged input."


def test_different_input_vector_lengths():
    """Test if correct error is raised."""
    vector1 = numpy.array([0, 0, 0, 0])
    vector2 = numpy.array([1, 1, 1, 1, 1])

    with pytest.raises(AssertionError) as msg:
        _ = cosine_similarity(vector1, vector2)

    expected_message = "Input vector must have same shape."
    assert expected_message == str(msg.value), "Expected particular error message."
