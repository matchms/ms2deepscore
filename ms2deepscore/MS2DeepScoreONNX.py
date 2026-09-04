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
from ms2deepscore.models import SiameseSpectralModelONNX
from .vector_operations import cosine_similarity, cosine_similarity_matrix


class MS2DeepScoreONNX(BaseSimilarity):
    """Calculate MS2DeepScore similarity scores between a reference and a query.

    Using a trained model, binned spectra will be converted into spectrum
    vectors using a deep neural network. The MS2DeepScoreONNX similarity is then
    the cosine similarity score between two spectrum vectors.

    Example code to calculate MS2DeepScoreONNX similarities between query and reference
    spectra:

    .. code-block:: python

        from matchms import calculate_scores
        from matchms.importing import load_spectra
        from ms2deepscore import MS2DeepScoreONNX
        from ms2deepscore.models import SiameseSpectralModelONNX

        # Import data
        references = list(load_spectra("abc.mgf"))
        queries = list(load_spectra("xyz.mgf"))

        # Load pretrained model
        model = SiameseSpectralModelONNX("data/ms2deepscore_model.onnx")

        similarity_measure = MS2DeepScoreONNX(model)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)


    """

    score_fields = ("score",)

    def __init__(self, model: SiameseSpectralModelONNX, progress_bar: bool = True):
        """

        Parameters
        ----------
        model:
            Expected input is a SiameseModelONNX with attached settings that has been trained on the desired set of spectra.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar. Default is False.
        """
        self.model = model
        self.output_vector_dim = self.model.model_settings.embedding_dim
        self.progress_bar = progress_bar

    def get_embedding_array(self, spectra, progress_bar: bool | None = None) -> np.ndarray:
        show_progress = self.progress_bar if progress_bar is None else progress_bar
        return self.model.compute_embedding_array(spectra, progress_bar=show_progress)

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
        embeddings = self.get_embedding_array([reference, query])
        return cosine_similarity(embeddings[0, :], embeddings[1, :])

    def _matrix_numpy(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        *,
        is_symmetric: bool,
        progress_bar: bool,
    ) -> np.ndarray:
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
        progress_bar:
            When True a progress bar is shown. Default is True.
        """
        embeddings_reference = self.get_embedding_array(
            references, progress_bar=progress_bar
        )
        if is_symmetric:
            embeddings_query = embeddings_reference
        else:
            embeddings_query = self.get_embedding_array(
                queries, progress_bar=progress_bar
            )
        return cosine_similarity_matrix(embeddings_reference, embeddings_query)

    if MATCHMS_V1_API:

        def matrix(
            self,
            spectra_1: List[Spectrum],
            spectra_2: List[Spectrum] | None = None,
            score_fields=None,
            progress_bar: bool = True,
        ):
            normalize_score_fields(score_fields, self.score_fields)
            is_symmetric = spectra_2 is None or spectra_2 is spectra_1
            queries = spectra_1 if spectra_2 is None else spectra_2
            score_matrix = self._matrix_numpy(
                spectra_1,
                queries,
                is_symmetric=is_symmetric,
                progress_bar=progress_bar,
            )
            return as_matchms_scores({"score": score_matrix})

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
                raise NotImplementedError("MS2DeepScoreONNX currently supports only array_type='numpy'.")
            if is_symmetric:
                assert_legacy_symmetric_inputs(references, queries)
            return self._matrix_numpy(
                references,
                queries,
                is_symmetric=is_symmetric,
                progress_bar=progress_bar,
            )
