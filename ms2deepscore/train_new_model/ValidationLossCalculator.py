import numpy as np
from matchms.similarity.vector_similarity_functions import \
    cosine_similarity_matrix
from ms2deepscore.benchmarking.calculate_scores_for_validation import \
    get_tanimoto_score_between_spectra
from ms2deepscore.models.loss_functions import bin_dependent_losses
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      compute_embedding_array)
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.utils import remove_diagonal


class ValidationLossCalculator:
    def __init__(self,
                 val_spectrums,
                 settings: SettingsMS2Deepscore):
        self.val_spectrums = select_spectra_per_inchikey(
            val_spectrums,
            settings.random_seed,
            settings.val_spectra_per_inchikey
            )
        self.score_bins = settings.same_prob_bins
        self.settings = settings
        tanimoto_scores = get_tanimoto_score_between_spectra(self.val_spectrums, self.val_spectrums,
                                                             self.settings.fingerprint_type,
                                                             self.settings.fingerprint_nbits,
                                                             )

        # Remove comparisons against themselves
        self.target_scores = remove_diagonal(tanimoto_scores)

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
                                            loss_types=loss_types,
                                            weighting_factor=self.settings.weighting_factor,
                                            )
        average_losses = {}
        binned_losses = {}
        for loss_type in loss_types:
            average_losses[loss_type] = np.mean(losses[loss_type])
            binned_losses[loss_type] = losses[loss_type]
        return average_losses, binned_losses


def select_spectra_per_inchikey(spectra,
                                random_seed,
                                spectra_per_inchikey: int = 1):
    """Pick spectra_per_inchikey spectra for every unique inchikey14 (when possible).
    """
    if spectra_per_inchikey < 1:
        raise ValueError
    inchikeys14_array = np.array([s.get("inchikey")[:14] for s in spectra])
    unique_inchikeys = np.unique(inchikeys14_array)
    rng = np.random.default_rng(seed=random_seed)
    selected_spectra = []
    for inchikey in unique_inchikeys:
        matching_spectra_idx = np.where(inchikeys14_array == inchikey)[0]
        if len(matching_spectra_idx) == 0:
            raise ValueError("Expected at least one spectrum per inchikey")
        selected_spectrum_ids = []
        for i in range(int(spectra_per_inchikey//len(matching_spectra_idx))):
            selected_spectrum_ids.extend(list(matching_spectra_idx))
        additional_spectrum_ids = rng.choice(matching_spectra_idx, spectra_per_inchikey%len(matching_spectra_idx),
                                             replace=False)
        selected_spectrum_ids.extend(additional_spectrum_ids)
        selected_spectra.extend([spectra[i] for i in selected_spectrum_ids])
    return selected_spectra
