import numpy as np
from matchms.similarity.vector_similarity_functions import \
    cosine_similarity_matrix
from ms2deepscore.benchmarking.calculate_scores_for_validation import \
    get_tanimoto_score_between_spectra
from ms2deepscore.models.loss_functions import bin_dependent_losses
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      compute_embedding_array)
from ms2deepscore.benchmarking.select_spectrum_pairs_for_visualization import remove_diagonal


class ValidationLossCalculator:
    def __init__(self,
                 val_spectrums,
                 score_bins=np.array([(x / 10, x / 10 + 0.1) for x in range(0, 10)]),
                 random_seed=42):
        self.val_spectrums = select_one_spectrum_per_inchikey(val_spectrums, random_seed)
        tanimoto_scores = get_tanimoto_score_between_spectra(self.val_spectrums, self.val_spectrums)
        # Remove comparisons against themselves
        self.target_scores = remove_diagonal(tanimoto_scores)
        self.score_bins = score_bins

    def compute_binned_validation_loss(self,
                                       model: SiameseSpectralModel,
                                       loss_types):
        """Benchmark the model against a validation set.
        """
        embeddings = compute_embedding_array(model, self.val_spectrums)
        ms2ds_scores = cosine_similarity_matrix(embeddings, embeddings)
        _, _, losses = bin_dependent_losses(remove_diagonal(ms2ds_scores),
                                            self.target_scores,
                                            self.score_bins,
                                            loss_types=loss_types
                                            )
        average_losses = {}
        for loss_type in loss_types:
            average_losses[loss_type] = np.mean(losses[loss_type])
        return average_losses


def select_one_spectrum_per_inchikey(spectra,
                                     random_seed):
    inchikeys14_array = np.array([s.get("inchikey")[:14] for s in spectra])
    unique_inchikeys = np.unique(inchikeys14_array)
    np.random.seed(random_seed)
    selected_spectra = []
    for inchikey in unique_inchikeys:
        matching_spectra_idx = np.where(inchikeys14_array == inchikey)[0]
        specrum_id = np.random.choice(matching_spectra_idx)
        selected_spectra.append(spectra[specrum_id])
    return selected_spectra