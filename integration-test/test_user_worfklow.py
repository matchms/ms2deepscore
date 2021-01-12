import os
import numpy as np
import pandas as pd
import pytest
from matchms import calculate_scores

from matchms.importing import load_from_mgf

from matchms import Spectrum
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllSpectrums


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
    """Test if a typical user workflow."""

    # Load and process spectrums
    spectrums = load_process_spectrums()
    assert spectrums[1].get("inchikey") == 'BBXXLROWFHWFQY-UHFFFAOYSA-N', \
        "Expected different metadata/spectrum"

    # Create binned spectrums
    ms2ds_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
    spectrums_binned = ms2ds_binner.fit_transform(spectrums)

    assert len(ms2ds_binner.known_bins) == 543, "Expected differnt number of known binned peaks"

    score_array, inchikey_mapping = get_reference_scores()

    # train model
    dimension = len(ms2ds_binner.known_bins)
    same_prob_bins = [(0, 0.5), (0.5, 1)]
    spectrum_ids = list(np.arange(0, len(spectrums)))

    # Create generator
    test_generator = DataGeneratorAllSpectrums(spectrums_binned, spectrum_ids, score_array,
                                               inchikey_mapping,
                                               dim=dimension,
                                               same_prob_bins=same_prob_bins)

    # Create Siamese model
    model = SiameseModel(input_dim=dimension, base_dims=(200, 200, 200), embedding_dim=200,
                         dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.summary()
    #x, y = zip(*test_generator)
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)


    # or: load model

    # calculate similarities
