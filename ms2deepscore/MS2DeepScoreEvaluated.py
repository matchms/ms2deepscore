from typing import List

import numpy as np
from matchms import Spectrum

from ms2deepscore.matchms_compat import (
    MATCHMS_V1_API,
    as_matchms_scores,
    normalize_score_fields,
    assert_legacy_symmetric_inputs,
)
if MATCHMS_V1_API:
    from matchms.similarity.base_similarity import BaseSimilarity
else:
    from matchms.similarity.BaseSimilarity import BaseSimilarity
from ms2deepscore.models.LinearEmbeddingEvaluation import compute_error_predictions
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.vector_operations import cosine_similarity, cosine_similarity_matrix


class MS2DeepScoreEvaluated(BaseSimilarity):
    """MS2DeepScore plus a predicted absolute-error field.
    
    Using a trained model, binned spectrums will be converted into spectrum
    vectors using a deep neural network. The MS2DeepScore similarity is then
    the cosine similarity score between two spectrum vectors.

    Example code to calcualte MS2DeepScore similarities between query and reference
    spectrums:

    # TODO: update code example to matchms 1.0

    .. code-block:: python

        from matchms import calculate_scores()
        from matchms.importing import load_from_json
        from ms2deepscore import MS2DeepScoreEvaluated
        from ms2deepscore.models import load_model, load_linear_model, load_embedding_evaluator

        # Import data
        references = load_from_json("abc.json")
        queries = load_from_json("xyz.json")

        # Load pretrained model
        model = load_model("model_file_123.pt")
        embedding_evaluator = load_embedding_evaluator("embedding_evaluator_123.pt")
        score_evaluator = load_linear_model("score_evaluator_123.json")

        similarity_measure = MS2DeepScoreEvaluated(model, embedding_evaluator, score_evaluator)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)

    """

    score_datatype = np.dtype(
        [("score", np.float32), ("predicted_absolute_error", np.float32)]
    )
    score_fields = ("score", "predicted_absolute_error")

    def __init__(
        self,
        model: SiameseSpectralModel,
        embedding_evaluator,
        score_evaluator,
        progress_bar: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.embedding_evaluator = embedding_evaluator
        self.embedding_evaluator.eval()
        self.score_evaluator = score_evaluator
        self.output_vector_dim = self.model.model_settings.embedding_dim
        self.progress_bar = progress_bar

    def get_embedding_array(
        self,
        spectra,
        datatype="numpy",
        batch_size=1024,
        progress_bar: bool | None = None,
    ):
        show_progress = self.progress_bar if progress_bar is None else progress_bar
        return self.model.compute_embedding_array(
            spectra,
            datatype=datatype,
            batch_size=batch_size,
            progress_bar=show_progress,
        )

    def get_embedding_evaluations(self, embeddings):
        """Compute predicted embedding RMSE values."""
        predicted_mse = self.embedding_evaluator(embeddings)
        predicted_mse[predicted_mse < 0] = 0
        return predicted_mse**0.5

    def get_score_evaluations(self, predicted_mse1, predicted_mse2):
        return compute_error_predictions(predicted_mse1, predicted_mse2, self.score_evaluator)

    def pair(self, reference: Spectrum, query: Spectrum):
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
        embeddings = self.get_embedding_array(
            [reference, query], datatype="pytorch"
        )
        evaluations = (
            self.get_embedding_evaluations(
                embeddings.reshape(-1, 1, self.output_vector_dim)
            )
            .detach()
            .cpu()
            .numpy()
        )
        score = cosine_similarity(
            embeddings[0, :].detach().cpu().numpy(),
            embeddings[1, :].detach().cpu().numpy(),
        )
        score_predicted_ae = self.get_score_evaluations(
            evaluations[0:1], evaluations[1:2]
        )[0, 0]
        return np.asarray(
            (float(score), float(score_predicted_ae)), dtype=self.score_datatype
        )

    def _matrix_components(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        *,
        is_symmetric: bool,
        progress_bar: bool,
        requested_fields: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        """Calculate the MS2DeepScore similarities between all references and queries.

        Parameters
        ----------
        references:
            Reference spectrum.
        queries:
            Query spectrum.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
        """
        embeddings_reference = self.get_embedding_array(
            references, datatype="pytorch", progress_bar=progress_bar
        )
        if is_symmetric:
            embeddings_query = embeddings_reference
        else:
            embeddings_query = self.get_embedding_array(
                queries, datatype="pytorch", progress_bar=progress_bar
            )

        result: dict[str, np.ndarray] = {}

        if "score" in requested_fields:
            result["score"] = cosine_similarity_matrix(
                embeddings_reference.detach().cpu().numpy(),
                embeddings_query.detach().cpu().numpy(),
            ).astype(np.float32, copy=False)

        if "predicted_absolute_error" in requested_fields:
            embeddings_ref_mse = (
                self.get_embedding_evaluations(
                    embeddings_reference.reshape(-1, 1, self.output_vector_dim)
                )
                .detach()
                .cpu()
                .numpy()
            )
            if is_symmetric:
                embeddings_query_mse = embeddings_ref_mse
            else:
                embeddings_query_mse = (
                    self.get_embedding_evaluations(
                        embeddings_query.reshape(-1, 1, self.output_vector_dim)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            result["predicted_absolute_error"] = self.get_score_evaluations(
                embeddings_ref_mse, embeddings_query_mse
            ).astype(np.float32, copy=False)

        return result

    if MATCHMS_V1_API:
        def matrix(
            self,
            spectra_1: List[Spectrum],
            spectra_2: List[Spectrum] | None = None,
            score_fields=None,
            progress_bar: bool = True,
        ):
            requested_fields = normalize_score_fields(score_fields, self.score_fields)
            is_symmetric = spectra_2 is None or spectra_2 is spectra_1
            queries = spectra_1 if spectra_2 is None else spectra_2
            score_arrays = self._matrix_components(
                spectra_1,
                queries,
                is_symmetric=is_symmetric,
                progress_bar=progress_bar,
                requested_fields=requested_fields,
            )
            return as_matchms_scores(score_arrays)

    else:
        def matrix(
            self,
            references: List[Spectrum],
            queries: List[Spectrum],
            array_type: str = "numpy",
            is_symmetric: bool = False,
            progress_bar: bool = True,
        ) -> np.ndarray:
            if array_type != "numpy":
                raise NotImplementedError(
                    "MS2DeepScoreEvaluated currently supports only array_type='numpy'."
                )
            if is_symmetric:
                assert_legacy_symmetric_inputs(references, queries)
            components = self._matrix_components(
                references,
                queries,
                is_symmetric=is_symmetric,
                progress_bar=progress_bar,
                requested_fields=self.score_fields,
            )
            similarities = np.empty(
                components["score"].shape, dtype=self.score_datatype
            )
            similarities["score"] = components["score"]
            similarities["predicted_absolute_error"] = components[
                "predicted_absolute_error"
            ]
            return similarities
