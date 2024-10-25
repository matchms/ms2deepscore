from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from tests.create_test_spectra import pesticides_test_spectra, TEST_RESOURCES_PATH


def test_calculate_scores_between_all_ionmodes():
    """Test that all dfs are created and have the correct shape"""
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:40]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[40:]]
    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.pt"
    scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(model_file, positive_mode_spectra,
                                                                    negative_mode_spectra)

    # Check that all dataframes are the correct shape
    nr_of_unique_pos_inchikeys = len(set([s.get("inchikey")[:14] for s in positive_mode_spectra]))
    nr_of_unique_neg_inchikeys = len(set([s.get("inchikey")[:14] for s in negative_mode_spectra]))

    assert scores_between_all_ionmodes.neg_vs_neg_scores.tanimoto_df.shape == (nr_of_unique_neg_inchikeys, nr_of_unique_neg_inchikeys)
    assert scores_between_all_ionmodes.pos_vs_pos_scores.tanimoto_df.shape == (nr_of_unique_pos_inchikeys, nr_of_unique_pos_inchikeys)
    assert scores_between_all_ionmodes.pos_vs_neg_scores.tanimoto_df.shape == (nr_of_unique_pos_inchikeys, nr_of_unique_neg_inchikeys)

    assert scores_between_all_ionmodes.neg_vs_neg_scores.predictions_df.shape == (len(negative_mode_spectra), len(negative_mode_spectra))
    assert scores_between_all_ionmodes.pos_vs_pos_scores.predictions_df.shape == (len(positive_mode_spectra), len(positive_mode_spectra))
    assert scores_between_all_ionmodes.pos_vs_neg_scores.predictions_df.shape == (len(positive_mode_spectra), len(negative_mode_spectra))
