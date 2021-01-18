from typing import List
from typing import Union
import numpy as np
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm import tqdm

from ms2deepscore.vector_operations import cosine_similarity
from ms2deepscore.vector_operations import cosine_similarity_matrix
from ms2deepscore import BinnedSpectrum
from ms2deepscore import SpectrumBinner
from ms2deepscore.models import SiameseModel


class MS2DeepScore(BaseSimilarity):
    """Calculate MS2DeepScore similarity scores between a reference and a query.

    Using a trained model, binned spectrums will be converted into spectrum
    vectors using a deep neural network. The MS2DeepScore similarity is then
    the cosine similarity score between two spectrum vectors.

    Example code to calcualte MS2DeepScore similarities between query and reference
    spectrums:

    .. code-block:: python

        from matchms import calculate_scores()
        from matchms.importing import load_from_json
        from ms2deepscore import MS2DeepScore
        from ms2deepscore.models import load_ms2ds_model

        references = load_from_json("abc.json")
        queries = load_from_json("xyz.json")

        model = ... TODO: implement such a function
        binned_references = model.spectrum_binner.transform(references)
        binned_queries = model.spectrum_binner.transform(queries)

        similarity_measure = MS2DeepScore(model)
        scores = calculate_scores(binned_references, binned_queries, similarity_measure)


    """
    def __init__(self, model, progress_bar: bool = False):
        """

        Parameters
        ----------
        model:
            Expected input is a SiameseModel that has been trained on
            the desired set of spectra.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
            Default is False.
        """
        self.model = model
        self.input_vector_dim = self.model.base.input_shape[1]  # TODO: later maybe also check against SpectrumBinner
        self.output_vector_dim = self.model.base.output_shape[1]
        self.disable_progress_bar = not progress_bar

    def _create_input_vectors(self, binned_spectrum: BinnedSpectrum):
        """Creates input vector for model.base based on binned peaks and intensities"""
        X = np.zeros((1, self.input_vector_dim))

        idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
        values = np.array([x for x in binned_spectrum.binned_peaks.values()])
        X[0, idx] = values
        return X

    def pair(self, reference: BinnedSpectrum, query: BinnedSpectrum) -> float:
        """Calculate the MS2DeepScore similaritiy between a reference and a query.

        Parameters
        ----------
        reference:
            Reference binned spectrum.
        query:
            Query binned spectrum.

        Returns
        -------
        ms2ds_similarity
            MS2DeepScore similarity score.
        """
        reference_vector = self.model.base.predict(self._create_input_vectors(reference))
        query_vector = self.model.base.predict(self._create_input_vectors(query))

        return cosine_similarity(reference_vector[0, :], query_vector[0, :])

    def matrix(self, references: List[BinnedSpectrum], queries: List[BinnedSpectrum],
               is_symmetric: bool = False) -> np.ndarray:
        """Calculate the MS2DeepScore similarities between all references and queries.

        Parameters
        ----------
        references:
            Reference spectrum documents.
        queries:
            Query spectrum documents.
        is_symmetric:
            Set to True if references == queries to speed up calculation about 2x.
            Uses the fact that in this case score[i, j] = score[j, i]. Default is False.

        Returns
        -------
        ms2ds_similarity
            Array of MS2DeepScore similarity scores.
        """
        n_rows = len(references)
        reference_vectors = np.empty((n_rows, self.output_vector_dim), dtype="float")
        for index_reference, reference in enumerate(tqdm(references, desc='Calculating vectors of reference spectrums', disable=self.disable_progress_bar)):
            reference_vectors[index_reference,
                              0:self.output_vector_dim] = self.model.base.predict(self._create_input_vectors(reference))
        n_cols = len(queries)
        query_vectors = np.empty((n_cols, self.output_vector_dim), dtype="float")
        for index_query, query in enumerate(tqdm(queries, desc='Calculating vectors of query spectrums', disable=self.disable_progress_bar)):
            query_vectors[index_query,
                          0:self.output_vector_dim] = self.model.base.predict(self._create_input_vectors(query))

        ms2ds_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)

        return ms2ds_similarity
