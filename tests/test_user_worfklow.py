from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from matchms.importing import load_from_mgf
from tensorflow import keras
from ms2deepscore import MS2DeepScore, SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def load_processed_spectrums():
    """Load processed spectrums from mgf file. For processing itself see matchms
    documentation."""
    spectrums_file = TEST_RESOURCES_PATH / "pesticides_processed.mgf"
    return list(load_from_mgf(spectrums_file.as_posix()))


def get_reference_scores():
    score_file = TEST_RESOURCES_PATH / "pesticides_tanimoto_scores.json"
    tanimoto_scores_df = pd.read_json(score_file)
    return tanimoto_scores_df


@pytest.mark.integtest
def test_user_workflow():
    """Test a typical user workflow from a mgf file to MS2DeepScore similarities."""

    # Load processed spectrums and reference scores (Tanimoto scores)
    spectrums = load_processed_spectrums()
    tanimoto_scores_df = get_reference_scores()
    # quick checks:
    assert spectrums[1].get("inchikey") == 'BBXXLROWFHWFQY-UHFFFAOYSA-N', \
        "Expected different metadata/spectrum"
    assert tanimoto_scores_df.shape == (45, 45), "Expected different shape for score array"

    # Create binned spectrums
    spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
    binned_spectrums = spectrum_binner.fit_transform(spectrums)
    assert spectrum_binner.d_bins == 0.99, "Expected differnt bin size"
    assert len(spectrum_binner.known_bins) == 543, "Expected differnt number of known binned peaks"

    # Create generator
    dimension = len(spectrum_binner.known_bins)
    same_prob_bins = [(0, 0.5), (0.5, 1)]
    test_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                               spectrum_binner=spectrum_binner,
                                               same_prob_bins=same_prob_bins)

    # Create (and train) a Siamese model
    model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200), embedding_dim=200, dropout_rate=0.2)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model.summary()
    model.fit(test_generator,
              validation_data=test_generator,
              epochs=2)

    # TODO: Add splitting data into training/validation/test
    # TODO: or load pretrained model instead

    # calculate similarities (pair)
    similarity_measure = MS2DeepScore(model)
    score = similarity_measure.pair(spectrums[0], spectrums[1])
    assert 0 < score < 1, "Expected score > 0 and < 1"
    assert isinstance(score, float), "Expected score to be float"

    # calculate similarities (matrix)
    scores = similarity_measure.matrix(spectrums[:10], spectrums[:10])
    assert scores.shape == (10, 10), "Expected different score array shape"
    assert np.allclose([scores[i, i] for i in range(10)], 1.0), "Expected diagonal values to be approx 1.0"
