from typing import List
import numpy as np
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm import tqdm

from .vector_operations import cosine_similarity
from .vector_operations import cosine_similarity_matrix
from .typing import BinnedSpectrumType


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
        from ms2deepscore.models import load_model

        # Import data
        references = load_from_json("abc.json")
        queries = load_from_json("xyz.json")

        # Load pretrained model
        model = load_model("model_file_123.hdf5")

        similarity_measure = MS2DeepScore(model)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)


    """
    def __init__(self, model, progress_bar: bool = True):
        """

        Parameters
        ----------
        model:
            Expected input is a SiameseModel that has been trained on
            the desired set of spectra. The model contains the keras deep neural
            network (model.model) as well as the used spectrum binner (model.spectrum_binner).
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
            Default is False.
        """
        self.model = model
        self.input_vector_dim = self.model.base.input_shape[1]  # TODO: later maybe also check against SpectrumBinner
        self.output_vector_dim = self.model.base.output_shape[1]
        self.progress_bar = progress_bar

    def _create_input_vector(self, binned_spectrum: BinnedSpectrumType):
        """Creates input vector for model.base based on binned peaks and intensities"""
        X = np.zeros((1, self.input_vector_dim))

        idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
        values = np.array(list(binned_spectrum.binned_peaks.values()))
        X[0, idx] = values
        return X

    def pair(self, reference: Spectrum, query: Spectrum) -> float:
        """Calculate the MS2DeepScore similaritiy between a reference and a query spectrum.

        Parameters
        ----------
        reference:
            Reference spectrum.
        query:
            Query spectrum.

        Returns
        -------
        ms2ds_similarity
            MS2DeepScore similarity score.
        """
        binned_reference = self.model.spectrum_binner.transform([reference])[0]
        binned_query = self.model.spectrum_binner.transform([query])[0]
        reference_vector = self.model.base.predict(self._create_input_vector(binned_reference))
        query_vector = self.model.base.predict(self._create_input_vector(binned_query))

        return cosine_similarity(reference_vector[0, :], query_vector[0, :])

    def matrix(self, references: List[Spectrum], queries: List[Spectrum],
               is_symmetric: bool = False) -> np.ndarray:
        """Calculate the MS2DeepScore similarities between all references and queries.

        Parameters
        ----------
        references:
            Reference spectrum.
        queries:
            Query spectrum.
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

        # Convert to binned spectrums
        binned_references = self.model.spectrum_binner.transform(references, progress_bar=self.progress_bar)
        binned_queries = self.model.spectrum_binner.transform(queries, progress_bar=self.progress_bar)

        for index_reference, reference in enumerate(tqdm(binned_references,
                                                         desc='Calculating vectors of reference spectrums',
                                                         disable=(not self.progress_bar))):
            reference_vectors[index_reference,
                              0:self.output_vector_dim] = self.model.base.predict(self._create_input_vector(reference))
        n_cols = len(queries)
        query_vectors = np.empty((n_cols, self.output_vector_dim), dtype="float")
        for index_query, query in enumerate(tqdm(binned_queries,
                                                 desc='Calculating vectors of query spectrums',
                                                 disable=(not self.progress_bar))):
            query_vectors[index_query,
                          0:self.output_vector_dim] = self.model.base.predict(self._create_input_vector(query))

        ms2ds_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)

        return ms2ds_similarity
