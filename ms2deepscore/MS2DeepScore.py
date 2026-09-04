from typing import List

import numpy as np
from matchms import Spectrum

from ms2deepscore.matchms_compat import (
    MATCHMS_V1_API,
    BaseSimilarity,
    as_matchms_scores,
    normalize_score_fields,
    assert_legacy_symmetric_inputs,
)
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from .vector_operations import cosine_similarity, cosine_similarity_matrix


class MS2DeepScore(BaseSimilarity):
    """Calculate MS2DeepScore similarity scores between spectra.

    Using a trained model, binned spectrums will be converted into spectrum
    vectors using a deep neural network. The MS2DeepScore similarity is then
    the cosine similarity score between two spectrum vectors.

    Example code to calcualte MS2DeepScore similarities between query and reference
    spectrums:

    .. code-block:: python

        from matchms import calculate_scores()
        from matchms.importing import load_ms2_dataset
        from ms2deepscore import MS2DeepScore
        from ms2deepscore.models import load_model

        # Import data
        references = load_ms2_dataset("abc.mgf")
        queries = load_ms2_dataset("xyz.mgf")

        # Load pretrained model
        model = load_model("model_file_123.pt")

        similarity_measure = MS2DeepScore(model)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)

    This implementation supports both the matchms <=0.33 matrix API (NumPy
    return values) and the matchms >=1.0 API (``matchms.Scores`` return values).
    """

    score_fields = ("score",)

    def __init__(self, model: SiameseSpectralModel, progress_bar: bool = True):
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
        self.model.eval()
        self.output_vector_dim = self.model.model_settings.embedding_dim
        self.progress_bar = progress_bar

    def get_embedding_array(
        self,
        spectra,
        datatype: str = "numpy",
        batch_size: int = 1024,
        progress_bar: bool | None = None,
    ) -> np.ndarray:
        """Calculate embeddings for a collection of spectra."""
        show_progress = self.progress_bar if progress_bar is None else progress_bar
        return self.model.compute_embedding_array(
            spectra,
            datatype=datatype,
            progress_bar=show_progress,
            batch_size=batch_size,
        )

    def pair(self, reference: Spectrum, query: Spectrum) -> float:
        """Calculate a single MS2DeepScore similarity.

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

        Returns
        -------
        ms2ds_similarity
            Array of MS2DeepScore similarity scores.
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
            """Return a matchms >=1.0 ``Scores`` object."""
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
            """Return a NumPy matrix for the matchms <=0.33 API."""
            if array_type != "numpy":
                raise NotImplementedError("MS2DeepScore currently supports only array_type='numpy'.")
            if is_symmetric:
                assert_legacy_symmetric_inputs(references, queries)
            return self._matrix_numpy(
                references,
                queries,
                is_symmetric=is_symmetric,
                progress_bar=progress_bar,
            )
