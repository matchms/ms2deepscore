from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from tests.test_user_worfklow import load_processed_spectrums, TEST_RESOURCES_PATH

def test_calculate_scores_between_all_ionmodes():
    spectrums = load_processed_spectrums()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.pt"
    scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(model_file, spectrums, spectrums)
    # todo add actual tests that scores are calculated correctly