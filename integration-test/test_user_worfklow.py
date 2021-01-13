import os
import numpy as np
import pandas as pd
import pytest
from tensorflow import keras
from matchms import calculate_scores
from matchms.importing import load_from_mgf
from matchms import Spectrum

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel


def load_process_spectrums():
    """Load processed spectrums from mgf file. For processing itself see matchms
    documentation."""
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "integration-test", "pesticides_processed.mgf")
    return list(load_from_mgf(spectrums_file))


def get_reference_scores():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    score_file = os.path.join(module_root, "integration-test", "pesticides_tanimoto_scores.json")
    tanimoto_scores = pd.read_json(score_file)
    return tanimoto_scores.values, tanimoto_scores.columns.to_numpy()


def test_user_workflow():
    """Test a typical user workflow from a mgf file to MS2DeepScore similarities."""

    # Load processed spectrums and reference scores (Tanimoto scores)
    spectrums = load_process_spectrums()
    score_array, inchikey_mapping = get_reference_scores()
    # quick checks:
    assert spectrums[1].get("inchikey") == 'BBXXLROWFHWFQY-UHFFFAOYSA-N', \
        "Expected different metadata/spectrum"
    assert score_array.shape == (45, 45), "Expected different shape for score array"

    # Create binned spectrums
    ms2ds_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
    spectrums_binned = ms2ds_binner.fit_transform(spectrums)
    assert ms2ds_binner.d_bins == 0.99, "Expected differnt bin size"
    assert len(ms2ds_binner.known_bins) == 543, "Expected differnt number of known binned peaks"

    # train model
    dimension = len(ms2ds_binner.known_bins)
    same_prob_bins = [(0, 0.5), (0.5, 1)]
    spectrum_ids = list(np.arange(0, len(spectrums)))

    # Create generator
    test_generator = DataGeneratorAllSpectrums(spectrums_binned, spectrum_ids, score_array,
                                               inchikey_mapping,
                                               dim=dimension,
                                               same_prob_bins=same_prob_bins)

    # Create (and train) a Siamese model
    model = SiameseModel(input_dim=dimension, base_dims=(200, 200, 200), embedding_dim=200,
                         dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.summary()
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)
    assert len(model.model.layers[2].layers) == len(model.base.layers) == 1, \
        "Expected different number of layers"

    # or: load model

    # calculate similarities
