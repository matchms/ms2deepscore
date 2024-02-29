from typing import List
import numpy as np
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from ms2deepscore.models.LinearEmbeddingEvaluation import \
    compute_error_predictions
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      compute_embedding_array)
from ms2deepscore.vector_operations import (cosine_similarity,
                                            cosine_similarity_matrix)


class MS2DeepScoreEvaluated(BaseSimilarity):
    """Calculate MS2DeepScore similarity scores between a reference and a query.

    Using a trained model, binned spectrums will be converted into spectrum
    vectors using a deep neural network. The MS2DeepScore similarity is then
    the cosine similarity score between two spectrum vectors.

    Example code to calcualte MS2DeepScore similarities between query and reference
    spectrums:

    .. code-block:: python

        from matchms import calculate_scores()
        from matchms.importing import load_from_json
        from ms2deepscore import MS2DeepScoreEvaluated
        from ms2deepscore.models import load_model, load_linear_model

        # Import data
        references = load_from_json("abc.json")
        queries = load_from_json("xyz.json")

        # Load pretrained model
        model = load_model("model_file_123.pt")
        embedding_evaluator = load_model("embedding_evaluator_123.pt")
        score_evaluator = load_linear_model("score_evaluator_123.json")

        similarity_measure = MS2DeepScoreEvaluated(model, embedding_evaluator, score_evaluator)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)

    """
    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    score_datatype = [("score", np.float32), ("predicted_absolute_error", np.float32)]

    def __init__(self, model: SiameseSpectralModel,
                 embedding_evaluator,
                 score_evaluator,
                 progress_bar: bool = True):
        """

        Parameters
        ----------
        model:
            Expected input is a SiameseModel that has been trained on
            the desired set of spectra.
        embedding_evaluator:
            Model trained on predicting the score quality (in form of MSE) based on an embedding.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
            Default is False.
        """
        self.model = model
        self.model.eval()
        self.embedding_evaluator = embedding_evaluator
        self.embedding_evaluator .eval()
        self.score_evaluator = score_evaluator
        self.output_vector_dim = self.model.model_settings.embedding_dim
        self.progress_bar = progress_bar

    def get_embedding_array(self, spectrums, datatype="numpy"):
        return compute_embedding_array(self.model, spectrums, datatype)

    def get_embedding_evaluations(self, embeddings):
        """Compute the RMSE.
        """
        predicted_mse = self.embedding_evaluator(embeddings)
        predicted_mse[predicted_mse < 0] = 0
        return predicted_mse ** 0.5

    def get_score_evaluations(self, predicted_mse1, predicted_mse2):
        return compute_error_predictions(predicted_mse1, predicted_mse2, self.score_evaluator)

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
        embedding_reference = self.get_embedding_array([reference], datatype="pytorch")
        embedding_query = self.get_embedding_array([query], datatype="pytorch")

        embedding_ref_mse = self.get_embedding_evaluations(embedding_reference.reshape(-1, 1, self.output_vector_dim)).detach().numpy()
        embedding_query_mse = self.get_embedding_evaluations(embedding_query.reshape(-1, 1, self.output_vector_dim)).detach().numpy()
        score = cosine_similarity(embedding_reference[0, :].detach().numpy(), embedding_query[0, :].detach().numpy())
        score_predicted_ae = self.score_evaluator.predict([[embedding_ref_mse[0][0], embedding_query_mse[0][0]]])
        return np.asarray((score, score_predicted_ae),
                          dtype=self.score_datatype)

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
        embeddings_reference = self.get_embedding_array(references, datatype="pytorch")
        if is_symmetric:
            assert np.all(references == queries), \
                "Expected references to be equal to queries for is_symmetric=True"
            embeddings_query = embeddings_reference
        else:
            embeddings_query = self.get_embedding_array(queries, datatype="pytorch")

        embeddings_ref_mse = self.get_embedding_evaluations(embeddings_reference.reshape(-1, 1, self.output_vector_dim)).detach().numpy()
        embeddings_query_mse = self.get_embedding_evaluations(embeddings_query.reshape(-1, 1, self.output_vector_dim)).detach().numpy()

        ms2ds_similarity = cosine_similarity_matrix(embeddings_reference.detach().numpy(), embeddings_query.detach().numpy())
        ms2ds_uncertainty = self.get_score_evaluations(embeddings_ref_mse, embeddings_query_mse)
        similarities=np.empty((ms2ds_similarity.shape[0],
                              ms2ds_similarity.shape[1]), dtype=self.score_datatype)
        similarities["score"] = ms2ds_similarity
        similarities["predicted_absolute_error"] = ms2ds_uncertainty
        return similarities
