from typing import List, Tuple
import numpy as np
import torch
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm import tqdm
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      compute_embedding_array)
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.vector_operations import (cosine_similarity_matrix,
                                            mean_pooling, median_pooling,
                                            percentile_pooling)


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
        model = load_model("model_file_123.pt")

        similarity_measure = MS2DeepScoreMonteCarlo(model, n_ensembles=10)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)


    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    score_datatype = [("score", np.float32), ("lower_bound", np.float32), ("upper_bound", np.float32)]

    def __init__(self, model,
                 n_ensembles: int = 10,
                 average_type: str = "median",
                 low=10,
                 high=90,
                 progress_bar: bool = True):
        """

        Parameters
        ----------
        model:
            Expected input is a SiameseModel that has been trained on
            the desired set of spectra. The model contains the pytorch deep neural
            network as well as the used model parameters.
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
        if self.model.encoder.dropout.p == 0:
            raise TypeError("Monte Carlo Dropout is not supposed to be used with a model where dropout-rate=0.")
        # Set model to train mode (switch on the Dropout layer(s))
        model.train()

        self.n_ensembles = n_ensembles
        assert average_type in ["median", "mean"], \
            "Non supported input for average_type. Must be 'median' or 'mean'."

        self.average_type = average_type
        self.output_vector_dim = self.model.model_settings.embedding_dim
        self.progress_bar = progress_bar
        self.low = low
        self.high = high

    def get_embedding_array(self, spectrums):
        return compute_embedding_array(self.model, spectrums)

    def get_embedding_ensemble(self, spectrum):
        return compute_embedding_ensemble(self.model, spectrum, self.n_ensembles)

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
        reference_vectors = self.get_embedding_ensemble(reference)
        query_vectors = self.get_embedding_ensemble(query)
        scores_ensemble = cosine_similarity_matrix(reference_vectors, query_vectors)
        if self.average_type == "median":
            average_similarity = np.median(scores_ensemble)
            lower_bound, upper_bound = np.percentile(scores_ensemble, [self.low, self.high])
        elif self.average_type == "mean":
            average_similarity = np.mean(scores_ensemble)
            lower_bound, upper_bound = np.percentile(scores_ensemble, [self.low, self.high])
        return np.asarray((average_similarity, lower_bound, upper_bound),
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
            percentile_low, percentile_high = percentile_pooling(ms2ds_similarity,
                                                                 self.n_ensembles,
                                                                 self.low, self.high)
        elif self.average_type == "mean":
            average_similarities = mean_pooling(ms2ds_similarity, self.n_ensembles)
            percentile_low, percentile_high = percentile_pooling(ms2ds_similarity,
                                                                 self.n_ensembles,
                                                                 self.low, self.high)

        similarities=np.empty((average_similarities.shape[0],
                              average_similarities.shape[1]), dtype=self.score_datatype)
        similarities["score"] = average_similarities
        similarities["lower_bound"] = percentile_low
        similarities["upper_bound"] = percentile_high
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
        for index_ref, reference in enumerate(
                tqdm(spectrum_list,
                     desc='Calculating vectors of reference spectrums',
                     disable=(not self.progress_bar))):
            embeddings = self.get_embedding_ensemble(reference)
            reference_vectors[index_ref * self.n_ensembles:(index_ref + 1) * self.n_ensembles,
                              0:self.output_vector_dim] = embeddings
        return reference_vectors


def compute_embedding_ensemble(model: SiameseSpectralModel,
                               spectrum,
                               n_ensembles):
    """Compute n_ensembles embeddings of one input spectrum.
    This assumes that dropout layers are active and create embedding variation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = tensorize_spectra(n_ensembles * [spectrum], model.model_settings)
    with torch.no_grad():
        embeddings = model.encoder(X[0].to(device), X[1].to(device)).cpu().detach().numpy()
    return embeddings
