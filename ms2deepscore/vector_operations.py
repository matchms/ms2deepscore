"""Performance optimized vector operations. Same as found in Spec2Vec
(https://github.com/iomega/spec2vec)."""

import numba
import numpy as np


@numba.njit
def cosine_similarity_matrix(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Fast implementation of cosine similarity between two arrays of vectors.

    For example:

    .. code-block:: python

        import numpy as np
        from spec2vec.vector_operations import cosine_similarity_matrix

        vectors_1 = np.array([[1, 1, 0, 0],
                              [1, 0, 1, 1]])
        vectors_2 = np.array([[0, 1, 1, 0],
                              [0, 0, 1, 1]])
        similarity_matrix = cosine_similarity_matrix(vectors_1, vectors_2)


    Parameters
    ----------
    vectors_1
        Numpy array of vectors. vectors_1.shape[0] is number of vectors, vectors_1.shape[1]
        is vector dimension.
    vectors_2
        Numpy array of vectors. vectors_2.shape[0] is number of vectors, vectors_2.shape[1]
        is vector dimension.
    """
    assert vectors_1.shape[1] == vectors_2.shape[1], "Input vectors must have same shape."
    vectors_1 = vectors_1.astype(np.float64)  # Numba dot only accepts float or complex arrays
    vectors_2 = vectors_2.astype(np.float64)
    norm_1 = np.sqrt(np.sum(vectors_1**2, axis=1))
    norm_2 = np.sqrt(np.sum(vectors_2**2, axis=1))
    for i in range(vectors_1.shape[0]):
        vectors_1[i] = vectors_1[i] / norm_1[i]
    for i in range(vectors_2.shape[0]):
        vectors_2[i] = vectors_2[i] / norm_2[i]
    return np.dot(vectors_1, vectors_2.T)


@numba.njit
def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> np.float64:
    """Calculate cosine similarity between two input vectors.

    For example:

    .. testcode::

        import numpy as np
        from spec2vec.vector_operations import cosine_similarity

        vector1 = np.array([1, 1, 0, 0])
        vector2 = np.array([1, 1, 1, 1])
        print("Cosine similarity: {:.3f}".format(cosine_similarity(vector1, vector2)))

    Should output

    .. testoutput::

        Cosine similarity: 0.707

    Parameters
    ----------
    vector1
        Input vector. Can be array of integers or floats.
    vector2
        Input vector. Can be array of integers or floats.
    """
    assert vector1.shape[0] == vector2.shape[0], "Input vector must have same shape."
    prod12 = 0
    prod11 = 0
    prod22 = 0
    for i in range(vector1.shape[0]):
        prod12 += vector1[i] * vector2[i]
        prod11 += vector1[i] * vector1[i]
        prod22 += vector2[i] * vector2[i]
    cosine_score = 0
    if prod11 != 0 and prod22 != 0:
        cosine_score = prod12 / np.sqrt(prod11 * prod22)
    return np.float64(cosine_score)


@numba.njit(fastmath=True)
def mean_pooling(scores_ensemble: np.ndarray, n_ensembles: int) -> np.ndarray:
    """Do mean pooling on an ensemble of scores."""
    dim_0 = int(scores_ensemble.shape[0]/n_ensembles)
    dim_1 = int(scores_ensemble.shape[1]/n_ensembles)
    scores_pooled = np.zeros((dim_0, dim_1))

    for i in range(dim_0):
        for j in range(dim_1):
            scores_pooled[i, j] = np.mean(scores_ensemble[i*n_ensembles:(i+1)*n_ensembles,
                                                          j*n_ensembles:(j+1)*n_ensembles])
    return scores_pooled


@numba.njit(fastmath=True)
def median_pooling(scores_ensemble: np.ndarray, n_ensembles: int) -> np.ndarray:
    """Do median pooling on an ensemble of scores."""
    dim_0 = int(scores_ensemble.shape[0]/n_ensembles)
    dim_1 = int(scores_ensemble.shape[1]/n_ensembles)
    scores_pooled = np.zeros((dim_0, dim_1))

    for i in range(dim_0):
        for j in range(dim_1):
            scores_pooled[i, j] = np.median(scores_ensemble[i*n_ensembles:(i+1)*n_ensembles,
                                                            j*n_ensembles:(j+1)*n_ensembles])
    return scores_pooled


@numba.njit(fastmath=True)
def std_pooling(scores_ensemble: np.ndarray, n_ensembles: int) -> np.ndarray:
    """Do standard deviation pooling on an ensemble of scores."""
    dim_0 = int(scores_ensemble.shape[0]/n_ensembles)
    dim_1 = int(scores_ensemble.shape[1]/n_ensembles)
    scores_pooled = np.zeros((dim_0, dim_1))

    for i in range(dim_0):
        for j in range(dim_1):
            scores_pooled[i, j] = np.std(scores_ensemble[i*n_ensembles:(i+1)*n_ensembles,
                                                         j*n_ensembles:(j+1)*n_ensembles])
    return scores_pooled


@numba.njit(fastmath=True)
def iqr_pooling(scores_ensemble: np.ndarray, n_ensembles: int) -> np.ndarray:
    """Do interquartile range (IQR) pooling on an ensemble of scores."""
    dim_0 = int(scores_ensemble.shape[0]/n_ensembles)
    dim_1 = int(scores_ensemble.shape[1]/n_ensembles)
    scores_pooled = np.zeros((dim_0, dim_1))

    for i in range(dim_0):
        for j in range(dim_1):
            q75, q25 = np.percentile(scores_ensemble[i*n_ensembles:(i+1)*n_ensembles,
                                                     j*n_ensembles:(j+1)*n_ensembles],
                                     [75 ,25])
            scores_pooled[i, j] = q75 - q25
    return scores_pooled
