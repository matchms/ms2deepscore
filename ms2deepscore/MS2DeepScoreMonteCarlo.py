from typing import List, Tuple
import numpy as np
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm import tqdm
from .typing import BinnedSpectrumType
from .vector_operations import (cosine_similarity_matrix, iqr_pooling,
                                mean_pooling, median_pooling, std_pooling)


class MS2DeepScoreMonteCarlo(BaseSimilarity):
    """Calculate MS2DeepScore ensemble similarity scores and STD between a reference
    and a query using Monte-Carlo Dropout

    Using a trained model, binned spectrums will be converted into spectrum
    vectors using a deep neural network. The MS2DeepScoreMonteCarlo similarity is then
    the median (or mean, depending on set `average_type`) of n_ensemble x n_ensemble
    cosine similarity score between two spectrum vectors.

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

        similarity_measure = MS2DeepScoreMonteCarlo(model, n_ensembles=10)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)


    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    score_datatype = [("score", np.float64), ("uncertainty", np.float64)]

    def __init__(self, model, n_ensembles: int = 10, average_type: str = "median",
                 progress_bar: bool = True):
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
        average_type:
            Choice between "median" and "mean" defining which type of averaging is used
            to compute the similarity score from all ensemble scores. Default is "median"
            in which case the uncertainty will be evaluate by the interquantile range (IQR).
            When using "mean" the standard deviation is taken as uncertainty measure.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
            Default is False.
        """
        self.model = model
        self.multi_inputs = (model.nr_of_additional_inputs > 0)
        self.n_ensembles = n_ensembles
        assert average_type in ["median", "mean"], \
            "Non supported input for average_type. Must be 'median' or 'mean'."
        self.average_type = average_type
        if self.multi_inputs:
            self.input_vector_dim = [self.model.base.input_shape[0][1], self.model.base.input_shape[1][1]]
        else:
            self.input_vector_dim = self.model.base.input_shape[1]
        self.output_vector_dim = self.model.base.output_shape[1]
        self.progress_bar = progress_bar
        self.partial_model = self._create_monte_carlo_base()

    def _create_input_vector(self, binned_spectrum: BinnedSpectrumType):
        """Creates input vector for model.base based on binned peaks and intensities"""
        X = np.zeros((1, self.input_vector_dim))

        idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
        values = np.array(list(binned_spectrum.binned_peaks.values()))
        X[0, idx] = values
        return X

    def _create_monte_carlo_base(self):
        """Rebuild base network with training=True"""
        base_dims = []
        dropout_rates = []
        for layer in self.model.base.layers:
            if "dense" in layer.name:
                base_dims.append(layer.units)
            if "dropout" in layer.name:
                dropout_rates.append(layer.rate)
        if np.unique(dropout_rates).shape[0] > 1:
            print(f"Found multiple different dropout rates. Selected 1st dropout rate: {dropout_rates[0]}")
        dropout_rate = dropout_rates[0]

        dropout_in_first_layer = ('dropout' in self.model.base.layers[3].name)

        # re-build base network with dropout layers always on
        base = self.model.get_base_model(base_dims=base_dims, embedding_dim=self.output_vector_dim, dropout_rate=dropout_rate,
                                         dropout_in_first_layer=dropout_in_first_layer, dropout_always_on=True)
        base.set_weights(self.model.base.get_weights())
        return base

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
        ms2ds_ensemble_similarity, ms2ds_ensemble_uncertainty
            Tuple of MS2DeepScore similarity score and uncertainty measure (STD/IQR).
        """
        binned_reference = self.model.spectrum_binner.transform([reference])[0]
        binned_query = self.model.spectrum_binner.transform([query])[0]
        reference_vectors = self.get_embedding_ensemble(binned_reference)
        query_vectors = self.get_embedding_ensemble(binned_query)
        scores_ensemble = cosine_similarity_matrix(reference_vectors, query_vectors)
        if self.average_type == "median":
            average_similarity = np.median(scores_ensemble)
            uncertainty = iqr_pooling(scores_ensemble, self.n_ensembles)[0, 0]
        elif self.average_type == "mean":
            average_similarity = np.mean(scores_ensemble)
            uncertainty = scores_ensemble.std()
        return np.asarray((average_similarity, uncertainty),
                          dtype=self.score_datatype)

    def matrix(self, references: List[Spectrum], queries: List[Spectrum],
               array_type: str = "numpy",
               is_symmetric: bool = False) -> np.ndarray:
        """Calculate the MS2DeepScoreMonteCarlo similarities between all references and queries.

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
        ms2ds_ensemble_similarity, ms2ds_ensemble_uncertainties
            Array of Tuples of MS2DeepScore similarity score and uncertainty measure (STD/IQR).
        """
        reference_vectors = self.calculate_vectors(references)
        if is_symmetric:
            assert np.all(references == queries), \
                "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = self.calculate_vectors(queries)

        ms2ds_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)
        if self.average_type == "median":
            average_similarities = median_pooling(ms2ds_similarity, self.n_ensembles)
            uncertainties = iqr_pooling(ms2ds_similarity, self.n_ensembles)
        elif self.average_type == "mean":
            average_similarities = mean_pooling(ms2ds_similarity, self.n_ensembles)
            uncertainties = std_pooling(ms2ds_similarity, self.n_ensembles)

        similarities=np.empty((average_similarities.shape[0],
                              average_similarities.shape[1]), dtype=self.score_datatype)
        similarities['score'] = average_similarities
        similarities['uncertainty'] = uncertainties
        return similarities

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
