from typing import List, Tuple
import numpy as np
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm import tqdm

from .vector_operations import cosine_similarity_matrix, mean_pooling, std_pooling
from .typing import BinnedSpectrumType


class MS2DeepScoreMonteCarlo(BaseSimilarity):
    """Calculate MS2DeepScore ensemble similarity scores and STD between a reference
    and a query using Monte-Carlo Dropout

    Using a trained model, binned spectrums will be converted into spectrum
    vectors using a deep neural network. The MS2DeepScoreMonteCarlo similarity is then
    the mean of n_ensemble x n_ensemble cosine similarity score between two spectrum
    vectors.

    Example code to calcualte MS2DeepScoreMonteCarlo similarities between query and
    reference spectrums:

    .. code-block:: python

        from matchms import calculate_scores()
        from matchms.importing import load_from_json
        from ms2deepscore import MS2DeepScoreMonteCarlo
        from ms2deepscore.models import load_model

        # Import data
        references = load_from_json("abc.json")
        queries = load_from_json("xyz.json")

        # Load pretrained model
        model = load_model("model_file_123.hdf5")

        similarity_measure = MS2DeepScoreMonteCarlo(model, n_ensembles=5)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)


    """
    def __init__(self, model, n_ensembles: int = 10, progress_bar: bool = True):
        """

        Parameters
        ----------
        model:
            Expected input is a SiameseModel that has been trained on
            the desired set of spectra. The model contains the keras deep neural
            network (model.model) as well as the used spectrum binner (model.spectrum_binner).
        n_ensembles
            Number of embeddings to create for every spectrum using Monte-Carlo Dropout.
            n_ensembles will lead to n_ensembles^2 scores of which the mean and STD will
            be taken.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
            Default is False.
        """
        self.model = model
        self.n_ensembles = n_ensembles
        self.input_vector_dim = self.model.base.input_shape[1]  # TODO: later maybe also check against SpectrumBinner
        self.output_vector_dim = self.model.base.output_shape[1]
        self.progress_bar = progress_bar
        self.partial_model = self._create_monte_carlo_encoder()

    def _create_input_vector(self, binned_spectrum: BinnedSpectrumType):
        """Creates input vector for model.base based on binned peaks and intensities"""
        X = np.zeros((1, self.input_vector_dim))

        idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
        values = np.array(list(binned_spectrum.binned_peaks.values()))
        X[0, idx] = values
        return X

    def _create_monte_carlo_encoder(self):
        """Rebuild base network with training=True"""
        dims = []
        for layer in self.model.base.layers:
            if "dense" in layer.name:
                dims.append(layer.units)
            if "dropout" in layer.name:
                dropout_rate = layer.rate

        # re-build encoder network with dropout layers always on
        encoder = self.model.get_base_model(input_dim=self.input_vector_dim,
                                            dims=dims,
                                            embedding_dim=self.output_vector_dim,
                                            dropout_rate=dropout_rate,
                                            dropout_always_on=True)
        encoder.set_weights(self.model.base.get_weights())
        return encoder

    def pair(self, reference: Spectrum, query: Spectrum) -> Tuple[float, float]:
        """Calculate the MS2DeepScoreMonteCarlo similaritiy between a reference
        and a query spectrum.

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
        reference_vectors = self.get_embedding_ensemble(binned_reference)
        query_vectors = self.get_embedding_ensemble(binned_query)
        scores_ensemble = cosine_similarity_matrix(reference_vectors, query_vectors)

        return scores_ensemble.mean(), scores_ensemble.std()

    def matrix(self, references: List[Spectrum], queries: List[Spectrum],
               is_symmetric: bool = False) -> np.ndarray:
        """Calculate the MS2DeepScoreMonteCarlo similarities between all references and queries.

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
        reference_vectors = self.calculate_vectors(references)
        if is_symmetric:
            assert np.all(references == queries), \
                "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = self.calculate_vectors(queries)

        ms2ds_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)
        return mean_pooling(ms2ds_similarity, self.n_ensembles), std_pooling(ms2ds_similarity, self.n_ensembles)

    def calculate_vectors(self, spectrum_list: List[Spectrum]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a list of vectors for all spectra

        parameters
        ----------
        spectrum_list:
            List of spectra for which the vector should be calculated
        """
        n_rows = len(spectrum_list) * self.n_ensembles
        reference_vectors = np.empty((n_rows, self.output_vector_dim), dtype="float")
        binned_spectrums = self.model.spectrum_binner.transform(spectrum_list,
                                                                progress_bar=self.progress_bar)
        for index_ref, reference in enumerate(
                tqdm(binned_spectrums,
                     desc='Calculating vectors of reference spectrums',
                     disable=(not self.progress_bar))):
            embeddings = self.get_embedding_ensemble(reference)
            reference_vectors[index_ref * self.n_ensembles:(index_ref + 1) * self.n_ensembles,
                              0:self.output_vector_dim] = embeddings
        return reference_vectors

    def get_embedding_ensemble(self, spectrum_binned):
        input_vector_array = np.tile(self._create_input_vector(spectrum_binned), (self.n_ensembles, 1))
        return self.partial_model.predict(input_vector_array)
