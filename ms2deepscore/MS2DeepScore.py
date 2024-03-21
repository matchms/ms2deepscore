from typing import List
import numpy as np
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      compute_embedding_array)
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
        model = load_model("model_file_123.pt")

        similarity_measure = MS2DeepScore(model)
        # Calculate scores and get matchms.Scores object
        scores = calculate_scores(references, queries, similarity_measure)


    """

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

    def get_embedding_array(self, spectrums):
        return compute_embedding_array(self.model, spectrums)

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
        embedding_reference = self.get_embedding_array([reference])
        embedding_query = self.get_embedding_array([query])
        return cosine_similarity(embedding_reference[0, :], embedding_query[0, :])

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
        embeddings_reference = self.get_embedding_array(references)
        if is_symmetric:
            assert np.all(references == queries), \
                "Expected references to be equal to queries for is_symmetric=True"
            embeddings_query = embeddings_reference
        else:
            embeddings_query = self.get_embedding_array(queries)

        ms2ds_similarity = cosine_similarity_matrix(embeddings_reference, embeddings_query)
        return ms2ds_similarity
