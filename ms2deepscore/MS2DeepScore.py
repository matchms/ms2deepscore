from typing import List
import numpy as np
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm import tqdm
from .typing import BinnedSpectrumType
from .vector_operations import cosine_similarity, cosine_similarity_matrix


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
        self.multi_inputs = (model.nr_of_additional_inputs > 0)
        if self.multi_inputs:
            self.input_vector_dim = [self.model.base.input_shape[0][1], self.model.base.input_shape[1][1]]
        else:
            self.input_vector_dim = self.model.base.input_shape[1]
        self.output_vector_dim = self.model.base.output_shape[1]
        self.progress_bar = progress_bar

    def _create_input_vector(self, binned_spectrum: BinnedSpectrumType):
        """Creates input vector for model.base based on binned peaks and intensities"""
        if self.multi_inputs:
            X = [np.zeros((1, i[1])) for i in self.model.base.input_shape]
            idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
            values = np.array(list(binned_spectrum.binned_peaks.values()))

            X[0][0, idx] = values
            X[1] = np.array([[float(value) for key, value in binned_spectrum.metadata.items() if (key != "inchikey")]])
        else:
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
               array_type: str = "numpy",
               is_symmetric: bool = False) -> np.ndarray:
        """Calculate the MS2DeepScore similarities between all references and queries.

        Parameters
        ----------
        references:
            Reference spectrum.
        queries:
            Query spectrum.
        array_type
            Specify the output array type. Can be "numpy" or "sparse".
            Currently, only "numpy" is supported and will return a numpy array.
            Future versions will include "sparse" as option to return a COO-sparse array.
        is_symmetric:
            Set to True if references == queries to speed up calculation about 2x.
            Uses the fact that in this case score[i, j] = score[j, i]. Default is False.

        Returns
        -------
        ms2ds_similarity
            Array of MS2DeepScore similarity scores.
        """
        reference_vectors = self.calculate_vectors(references)
        if is_symmetric:
            assert np.all(references == queries), \
                "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = self.calculate_vectors(queries)

        ms2ds_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)
        return ms2ds_similarity

    def calculate_vectors(self, spectrum_list: List[Spectrum]) -> np.ndarray:
        """Returns a list of vectors for all spectra

        parameters
        ----------
        spectrum_list:
            List of spectra for which the vector should be calculated
        """
        n_rows = len(spectrum_list)
        reference_vectors = np.empty(
            (n_rows, self.output_vector_dim), dtype="float")
        binned_spectrums = self.model.spectrum_binner.transform(spectrum_list, progress_bar=self.progress_bar)
        for index_reference, reference in enumerate(
                tqdm(binned_spectrums,
                     desc='Calculating vectors of reference spectrums',
                     disable=(not self.progress_bar))):
            reference_vectors[index_reference, 0:self.output_vector_dim] = \
                self.model.base.predict(self._create_input_vector(reference), verbose=0)
        return reference_vectors
