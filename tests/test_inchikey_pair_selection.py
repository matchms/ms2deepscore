from collections import Counter
import numpy as np
import pytest
from matchms import Spectrum

from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    select_inchi_for_unique_inchikeys,
    create_spectrum_pair_generator,
    compute_fingerprints_for_training,
)
from ms2deepscore.train_new_model import SpectrumPairGenerator
from tests.create_test_spectra import create_test_spectra
import ms2deepscore.train_new_model.inchikey_pair_selection as pair_selection_module
import ms2deepscore.train_new_model.pair_data_persistence as pair_data_persistence


def _make_training_settings(
    fingerprint_type="rdkit_binary",
    bins=None,
    batch_size=8,
    average_inchikey_sampling_count=10,
    max_pair_resampling=10,
    max_inchikey_sampling=280,
):
    if bins is None:
        bins = [
            (0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7),
            (0.5, 0.6), (0.4, 0.5), (0.3, 0.4), (-0.01, 0.3)
        ]
    return SettingsMS2Deepscore(
        same_prob_bins=np.array(bins, dtype="float32"),
        average_inchikey_sampling_count=average_inchikey_sampling_count,
        batch_size=batch_size,
        max_pair_resampling=max_pair_resampling,
        max_inchikey_sampling=max_inchikey_sampling,
        fingerprint_type=fingerprint_type,
        fingerprint_nbits=256,
    )


@pytest.fixture
def test_spectra():
    metadata = {
        "precursor_mz": 101.1,
        "inchikey": "ABCABCABCABCAB-nonsense",
        "inchi": "InChI=1/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1"
    }
    spectrum_1 = Spectrum(
        mz=np.array([100.]),
        intensities=np.array([0.7]),
        metadata=metadata
    )
    spectrum_2 = Spectrum(
        mz=np.array([90.]),
        intensities=np.array([0.4]),
        metadata=metadata
    )
    spectrum_3 = Spectrum(
        mz=np.array([90.]),
        intensities=np.array([0.4]),
        metadata=metadata
    )
    spectrum_4 = Spectrum(
        mz=np.array([90.]),
        intensities=np.array([0.4]),
        metadata={
            "inchikey": 14 * "X",
            "inchi": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"
        }
    )
    return [spectrum_1, spectrum_2, spectrum_3, spectrum_4]


def test_select_inchi_for_unique_inchikeys(test_spectra):
    test_spectra[2].set("inchikey", "ABCABCABCABCAB-nonsense2")
    test_spectra[3].set("inchikey", "ABCABCABCABCAB-nonsense3")
    spectrums_selected, inchikey14s = select_inchi_for_unique_inchikeys(test_spectra)
    assert inchikey14s == ["ABCABCABCABCAB"]
    assert spectrums_selected[0].get("inchi").startswith("InChI=1/C6H8O6/")


def test_select_inchi_for_unique_inchikeys_two_inchikeys(test_spectra):
    spectrums_selected, inchikey14s = select_inchi_for_unique_inchikeys(test_spectra)
    assert inchikey14s == ["ABCABCABCABCAB", "XXXXXXXXXXXXXX"]
    assert [s.get("inchi")[:15] for s in spectrums_selected] == [
        "InChI=1/C6H8O6/",
        "InChI=1S/C8H10N",
    ]


def test_SelectedInchikeyPairs_generator_with_shuffle():
    dummy_inchikey_pair_generator = SpectrumPairGenerator(
        [
            ("Inchikey0", "Inchikey1", 0.8),
            ("Inchikey0", "Inchikey2", 0.6),
            ("Inchikey2", "Inchikey1", 0.3),
            ("Inchikey2", "Inchikey2", 1.0),
        ],
        [
            Spectrum(mz=np.array([90.]), intensities=np.array([0.4]), metadata={"inchikey": "Inchikey0"}),
            Spectrum(mz=np.array([90.]), intensities=np.array([0.4]), metadata={"inchikey": "Inchikey1"}),
            Spectrum(mz=np.array([90.]), intensities=np.array([0.4]), metadata={"inchikey": "Inchikey2"}),
        ],
        True,
        0,
    )

    found_pairs = []
    for _ in range(len(dummy_inchikey_pair_generator)):
        spectrum_1, spectrum_2, score = next(dummy_inchikey_pair_generator)
        found_pairs.append((spectrum_1.get("inchikey"), spectrum_2.get("inchikey"), score))

    assert len(found_pairs) == len(dummy_inchikey_pair_generator.selected_inchikey_pairs)
    assert sorted(found_pairs) == sorted(dummy_inchikey_pair_generator.selected_inchikey_pairs)

    found_pairs = []
    for _ in range(len(dummy_inchikey_pair_generator)):
        spectrum_1, spectrum_2, score = next(dummy_inchikey_pair_generator)
        found_pairs.append((spectrum_1.get("inchikey"), spectrum_2.get("inchikey"), score))

    assert len(found_pairs) == len(dummy_inchikey_pair_generator.selected_inchikey_pairs)
    assert sorted(found_pairs) == sorted(dummy_inchikey_pair_generator.selected_inchikey_pairs)


def test_SelectedInchikeyPairs_generator_without_shuffle():
    dummy_inchikey_pair_generator = SpectrumPairGenerator(
        [
            ("Inchikey0", "Inchikey1", 0.8),
            ("Inchikey0", "Inchikey2", 0.6),
            ("Inchikey2", "Inchikey1", 0.3),
            ("Inchikey2", "Inchikey2", 1.0),
        ],
        [
            Spectrum(mz=np.array([90.]), intensities=np.array([0.4]), metadata={"inchikey": "Inchikey0"}),
            Spectrum(mz=np.array([90.]), intensities=np.array([0.4]), metadata={"inchikey": "Inchikey1"}),
            Spectrum(mz=np.array([90.]), intensities=np.array([0.4]), metadata={"inchikey": "Inchikey2"}),
        ],
        False,
        0,
    )

    for expected_pair in dummy_inchikey_pair_generator.selected_inchikey_pairs:
        spectrum_1, spectrum_2, score = next(dummy_inchikey_pair_generator)
        assert expected_pair == (spectrum_1.get("inchikey"), spectrum_2.get("inchikey"), score)


def check_correct_oversampling(selected_inchikey_pairs: SpectrumPairGenerator, max_resampling: int):
    pair_counts = Counter(selected_inchikey_pairs.selected_inchikey_pairs)
    for count in pair_counts.values():
        assert count <= max_resampling, "the resampling was done too frequently"


def check_balanced_scores_selecting_inchikey_pairs(selected_inchikey_pairs: SpectrumPairGenerator, score_bins):
    scores = selected_inchikey_pairs.get_scores()
    score_bins = np.array(score_bins, dtype="float32")
    score_bin_counts = {tuple(score_bin): 0 for score_bin in score_bins}
    for score in scores:
        for min_bound, max_bound in score_bins:
            if score > min_bound and score <= max_bound:
                score_bin_counts[(min_bound, max_bound)] += 1
    assert len(set(score_bin_counts.values())) == 1


@pytest.mark.parametrize("fingerprint_type", ["rdkit_binary", "rdkit_count"])
def test_compute_fingerprints_for_training_dense_types(fingerprint_type):
    spectrums = create_test_spectra(num_of_unique_inchikeys=10, num_of_spectra_per_inchikey=2)

    fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
        spectrums,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )

    assert isinstance(inchikeys14_unique, list)
    assert len(inchikeys14_unique) == 10
    assert len(fingerprints) == 10
    assert fingerprints.shape[0] == 10
    assert np.all(np.sum(fingerprints, axis=1) > 0)


@pytest.mark.parametrize("fingerprint_type", ["rdkit_binary_unfolded", "rdkit_count_unfolded"])
def test_compute_fingerprints_for_training_unfolded_types_return_expected_length(fingerprint_type):
    spectrums = create_test_spectra(num_of_unique_inchikeys=10, num_of_spectra_per_inchikey=2)

    fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
        spectrums,
        fingerprint_type=fingerprint_type,
        nbits=256,
    )

    assert isinstance(inchikeys14_unique, list)
    assert len(inchikeys14_unique) == 10
    assert len(fingerprints) == 10


@pytest.mark.parametrize(
    "fingerprint_type,bins,max_pair_resampling",
    [
        ("rdkit_binary", [(0.5, 0.75), (0.25, 0.5), (0.75, 1), (-0.000001, 0.25)], 1),
        ("rdkit_count",  [(0.5, 0.75), (0.25, 0.5), (0.75, 1), (-0.000001, 0.25)], 1),
        ("rdkit_binary_unfolded", [(-0.01, 1.0)], 1),
        ("rdkit_count_unfolded", [(-0.01, 1.0)], 1),
    ],
)
def test_select_compound_pairs_wrapper_no_resampling_supported_types(
    fingerprint_type, bins, max_pair_resampling
):
    spectrums = create_test_spectra(num_of_unique_inchikeys=26, num_of_spectra_per_inchikey=2)
    settings = SettingsMS2Deepscore(
        same_prob_bins=np.array(bins, dtype="float32"),
        average_inchikey_sampling_count=10,
        batch_size=8,
        max_pair_resampling=max_pair_resampling,
        fingerprint_type=fingerprint_type,
    )
    inchikey_pair_generator = create_spectrum_pair_generator(spectrums, settings)

    check_balanced_scores_selecting_inchikey_pairs(inchikey_pair_generator, bins)
    check_correct_oversampling(inchikey_pair_generator, max_pair_resampling)


@pytest.mark.parametrize(
    "fingerprint_type,bins,max_pair_resampling",
    [
        ("rdkit_binary", [(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
                          (0.4, 0.5), (0.3, 0.4), (-0.01, 0.3)], 10),
        ("rdkit_count", [(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
                         (0.4, 0.5), (0.3, 0.4), (-0.01, 0.3)], 10),
        ("rdkit_binary_unfolded", [(-0.01, 1.0)], 10),
        ("rdkit_count_unfolded", [(-0.01, 1.0)], 10),
    ],
)
def test_select_compound_pairs_wrapper_with_resampling_supported_types(
    fingerprint_type, bins, max_pair_resampling
):
    spectrums = create_test_spectra(num_of_unique_inchikeys=26, num_of_spectra_per_inchikey=1)
    settings = SettingsMS2Deepscore(
        same_prob_bins=np.array(bins, dtype="float32"),
        average_inchikey_sampling_count=10,
        batch_size=8,
        max_pair_resampling=max_pair_resampling,
        fingerprint_type=fingerprint_type,
    )
    inchikey_pair_generator = create_spectrum_pair_generator(spectrums, settings)

    check_balanced_scores_selecting_inchikey_pairs(inchikey_pair_generator, bins)
    check_correct_oversampling(inchikey_pair_generator, max_pair_resampling)


@pytest.mark.parametrize(
    "fingerprint_type,bins",
    [
        ("rdkit_binary", [(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
                          (0.4, 0.5), (0.3, 0.4), (-0.01, 0.3)]),
        ("rdkit_count", [(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
                         (0.4, 0.5), (0.3, 0.4), (-0.01, 0.3)]),
        ("rdkit_binary_unfolded", [(-0.01, 1.0)]),
        ("rdkit_count_unfolded", [(-0.01, 1.0)]),
    ],
)
def test_select_compound_pairs_wrapper_maximum_inchikey_count_supported_types(fingerprint_type, bins):
    spectrums = create_test_spectra(num_of_unique_inchikeys=26, num_of_spectra_per_inchikey=1)
    max_pair_resampling = 1000
    max_inchikey_sampling = 280
    settings = SettingsMS2Deepscore(
        same_prob_bins=np.array(bins, dtype="float32"),
        average_inchikey_sampling_count=200,
        batch_size=8,
        max_pair_resampling=max_pair_resampling,
        max_inchikey_sampling=max_inchikey_sampling,
        fingerprint_type=fingerprint_type,
    )
    inchikey_pair_generator = create_spectrum_pair_generator(spectrums, settings)

    highest_inchikey_count = max(inchikey_pair_generator.get_inchikey_counts().values())
    assert highest_inchikey_count <= max_inchikey_sampling + 1


# -----------------------------------------------------------------------------
# Tests for available-pair counting and explicit pair-data folders
# -----------------------------------------------------------------------------


def test_get_nr_of_available_pairs_in_bin_counts_unique_undirected_pairs():
    """Mirrored/repeated pairs count once, self-pairs count once, empty bins count zero."""
    selected_pairs_per_bin_matrix = np.array(
        [
            [
                [1, 1, 2],   # 0-1 twice, 0-2
                [0, 2, -1],  # mirrored 0-1, plus 1-2
                [0, 1, 2],   # mirrored 0-2 and 1-2, plus self-pair 2-2
            ],
            [
                [-1, -1, -1],
                [1, -1, -1],  # self-pair 1-1
                [-1, -1, -1],
            ],
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
        ],
        dtype=np.int32,
    )

    assert pair_selection_module.get_nr_of_available_pairs_in_bin(
        selected_pairs_per_bin_matrix
    ) == [4, 1, 0]


def test_get_nr_of_available_pairs_in_bin_does_not_require_symmetric_storage():
    """A pair stored in only one direction is still one available unique pair."""
    selected_pairs_per_bin_matrix = np.array(
        [
            [
                [2, -1],
                [-1, -1],
                [-1, -1],
            ]
        ],
        dtype=np.int32,
    )

    assert pair_selection_module.get_nr_of_available_pairs_in_bin(
        selected_pairs_per_bin_matrix
    ) == [1]


def _make_pair_folder_test_settings():
    return SettingsMS2Deepscore(
        same_prob_bins=np.array([(-0.01, 1.0)], dtype=np.float32),
        average_inchikey_sampling_count=2,
        batch_size=2,
        max_pair_resampling=5,
        max_inchikey_sampling=20,
        max_pairs_per_bin=2,
        fingerprint_type="rdkit_binary",
        fingerprint_nbits=64,
        random_seed=13,
        shuffle=False,
    )


def _tiny_candidate_data(inchikeys):
    n = len(inchikeys)
    pairs = np.full((1, n, 2), -1, dtype=np.int32)
    scores = np.zeros((1, n, 2), dtype=np.float32)
    for i in range(n):
        j = (i + 1) % n
        pairs[0, i, 0] = j
        scores[0, i, 0] = 0.5
    return pairs, scores


def _install_fake_pair_preparation(monkeypatch, spectra, selected_pairs, calls):
    inchikeys = sorted({s.get("inchikey")[:14] for s in spectra})
    candidate_pairs, candidate_scores = _tiny_candidate_data(inchikeys)

    def fake_fingerprints(*args, **kwargs):
        calls["fingerprints"] += 1
        return np.zeros((len(inchikeys), 8), dtype=np.float32), inchikeys

    def fake_candidates(*args, **kwargs):
        calls["candidates"] += 1
        return candidate_pairs.copy(), candidate_scores.copy()

    def fake_balancing(*args, **kwargs):
        calls["balancing"] += 1
        return np.zeros_like(candidate_pairs, dtype=np.int32)

    def fake_conversion(*args, **kwargs):
        calls["conversion"] += 1
        return [list(selected_pairs)]

    monkeypatch.setattr(
        pair_selection_module,
        "compute_fingerprints_for_training",
        fake_fingerprints,
    )
    monkeypatch.setattr(
        pair_selection_module,
        "compute_tanimoto_similarity_per_bin",
        fake_candidates,
    )
    monkeypatch.setattr(
        pair_selection_module,
        "balanced_selection_of_pairs_per_bin",
        fake_balancing,
    )
    monkeypatch.setattr(
        pair_selection_module,
        "convert_to_selected_pairs_list",
        fake_conversion,
    )

    return inchikeys, candidate_pairs, candidate_scores


def test_pair_data_folder_first_run_writes_candidate_data_and_final_schedule(
    tmp_path, monkeypatch
):
    spectra = create_test_spectra(
        num_of_unique_inchikeys=4,
        num_of_spectra_per_inchikey=1,
    )
    settings = _make_pair_folder_test_settings()
    inchikeys = sorted({s.get("inchikey")[:14] for s in spectra})
    selected_pairs = [
        (inchikeys[0], inchikeys[1], 0.25),
        (inchikeys[2], inchikeys[3], 0.75),
    ]
    calls = Counter()
    _install_fake_pair_preparation(monkeypatch, spectra, selected_pairs, calls)

    generator = create_spectrum_pair_generator(
        spectra,
        settings,
        pair_data_folder=tmp_path,
    )

    assert calls == Counter(
        fingerprints=1,
        candidates=1,
        balancing=1,
        conversion=1,
    )
    assert pair_data_persistence.candidate_data_exists(tmp_path)
    assert pair_data_persistence.selected_schedule_exists(tmp_path)
    assert (tmp_path / pair_data_persistence.MANIFEST_FILENAME).is_file()

    actual_pairs = list(generator.selected_inchikey_pairs)
    assert [pair[:2] for pair in actual_pairs] == [pair[:2] for pair in selected_pairs]
    np.testing.assert_allclose(
        [pair[2] for pair in actual_pairs],
        [pair[2] for pair in selected_pairs],
    )


def test_pair_data_folder_second_run_reuses_final_schedule_without_recomputation(
    tmp_path, monkeypatch
):
    spectra = create_test_spectra(4, 1)
    settings = _make_pair_folder_test_settings()
    inchikeys = sorted({s.get("inchikey")[:14] for s in spectra})
    selected_pairs = [
        (inchikeys[0], inchikeys[1], 0.25),
        (inchikeys[2], inchikeys[3], 0.75),
    ]
    calls = Counter()
    _install_fake_pair_preparation(monkeypatch, spectra, selected_pairs, calls)

    first_generator = create_spectrum_pair_generator(
        spectra,
        settings,
        pair_data_folder=tmp_path,
    )
    first_pairs = list(first_generator.selected_inchikey_pairs)

    def unexpected_call(*args, **kwargs):
        pytest.fail("Pair preparation should not run when a final schedule exists.")

    monkeypatch.setattr(pair_selection_module, "compute_fingerprints_for_training", unexpected_call)
    monkeypatch.setattr(pair_selection_module, "compute_tanimoto_similarity_per_bin", unexpected_call)
    monkeypatch.setattr(pair_selection_module, "balanced_selection_of_pairs_per_bin", unexpected_call)
    monkeypatch.setattr(pair_selection_module, "convert_to_selected_pairs_list", unexpected_call)

    second_generator = create_spectrum_pair_generator(
        spectra,
        settings,
        pair_data_folder=tmp_path,
    )
    second_pairs = list(second_generator.selected_inchikey_pairs)

    assert [pair[:2] for pair in second_pairs] == [pair[:2] for pair in first_pairs]
    np.testing.assert_allclose(
        [pair[2] for pair in second_pairs],
        [pair[2] for pair in first_pairs],
    )


def test_pair_data_folder_reuses_candidate_checkpoint_when_final_schedule_is_missing(
    tmp_path, monkeypatch
):
    spectra = create_test_spectra(4, 1)
    settings = _make_pair_folder_test_settings()
    inchikeys = sorted({s.get("inchikey")[:14] for s in spectra})
    candidate_pairs, candidate_scores = _tiny_candidate_data(inchikeys)
    selected_pairs = [
        (inchikeys[0], inchikeys[1], 0.4),
        (inchikeys[2], inchikeys[3], 0.6),
    ]

    manifest = pair_data_persistence.build_pair_data_manifest(
        kind="within_set",
        settings=settings,
        spectra_signature_1=pair_data_persistence.spectra_structure_signature(spectra),
    )
    pair_data_persistence.prepare_pair_data_folder(tmp_path, manifest)
    pair_data_persistence.save_candidate_pair_data(
        tmp_path,
        candidate_pairs,
        candidate_scores,
        inchikeys,
    )

    def unexpected_call(*args, **kwargs):
        pytest.fail("Fingerprint/candidate computation should be skipped when candidate data exists.")

    calls = Counter()

    def fake_balancing(*args, **kwargs):
        calls["balancing"] += 1
        return np.zeros_like(candidate_pairs, dtype=np.int32)

    def fake_conversion(*args, **kwargs):
        calls["conversion"] += 1
        return [selected_pairs]

    monkeypatch.setattr(pair_selection_module, "compute_fingerprints_for_training", unexpected_call)
    monkeypatch.setattr(pair_selection_module, "compute_tanimoto_similarity_per_bin", unexpected_call)
    monkeypatch.setattr(pair_selection_module, "balanced_selection_of_pairs_per_bin", fake_balancing)
    monkeypatch.setattr(pair_selection_module, "convert_to_selected_pairs_list", fake_conversion)

    generator = create_spectrum_pair_generator(
        spectra,
        settings,
        pair_data_folder=tmp_path,
    )

    assert calls == Counter(balancing=1, conversion=1)
    assert pair_data_persistence.selected_schedule_exists(tmp_path)
    assert len(generator) == len(selected_pairs)


def test_pair_data_folder_rejects_changed_pair_selection_settings_before_recomputation(
    tmp_path, monkeypatch
):
    spectra = create_test_spectra(4, 1)
    settings = _make_pair_folder_test_settings()

    manifest = pair_data_persistence.build_pair_data_manifest(
        kind="within_set",
        settings=settings,
        spectra_signature_1=pair_data_persistence.spectra_structure_signature(spectra),
    )
    pair_data_persistence.prepare_pair_data_folder(tmp_path, manifest)

    changed_settings = _make_pair_folder_test_settings()
    changed_settings.fingerprint_nbits = settings.fingerprint_nbits * 2

    def unexpected_call(*args, **kwargs):
        pytest.fail("Pair preparation should not start before manifest validation.")

    monkeypatch.setattr(pair_selection_module, "compute_fingerprints_for_training", unexpected_call)

    with pytest.raises(ValueError, match="different inputs/settings"):
        create_spectrum_pair_generator(
            spectra,
            changed_settings,
            pair_data_folder=tmp_path,
        )


def test_pair_data_folder_allows_model_only_setting_changes(
    tmp_path, monkeypatch
):
    spectra = create_test_spectra(4, 1)
    settings = _make_pair_folder_test_settings()
    inchikeys = sorted({s.get("inchikey")[:14] for s in spectra})
    selected_pairs = [(inchikeys[0], inchikeys[1], 0.5)]
    calls = Counter()
    _install_fake_pair_preparation(monkeypatch, spectra, selected_pairs, calls)

    create_spectrum_pair_generator(
        spectra,
        settings,
        pair_data_folder=tmp_path,
    )

    # These settings do not influence pair preparation and therefore must not
    # invalidate an explicitly persisted pair-data folder.
    settings.learning_rate *= 2
    settings.dropout_rate = 0.25
    settings.embedding_dim += 1

    def unexpected_call(*args, **kwargs):
        pytest.fail("Model-only setting changes should still reuse the final pair schedule.")

    monkeypatch.setattr(pair_selection_module, "compute_fingerprints_for_training", unexpected_call)
    monkeypatch.setattr(pair_selection_module, "compute_tanimoto_similarity_per_bin", unexpected_call)
    monkeypatch.setattr(pair_selection_module, "balanced_selection_of_pairs_per_bin", unexpected_call)
    monkeypatch.setattr(pair_selection_module, "convert_to_selected_pairs_list", unexpected_call)

    generator = create_spectrum_pair_generator(
        spectra,
        settings,
        pair_data_folder=tmp_path,
    )

    assert len(generator) == 1


def test_pair_data_folder_rejects_incomplete_selected_schedule(tmp_path):
    spectra = create_test_spectra(4, 1)
    settings = _make_pair_folder_test_settings()
    manifest = pair_data_persistence.build_pair_data_manifest(
        kind="within_set",
        settings=settings,
        spectra_signature_1=pair_data_persistence.spectra_structure_signature(spectra),
    )
    pair_data_persistence.prepare_pair_data_folder(tmp_path, manifest)

    # Simulate an interrupted/corrupt write: one schedule file exists, the others do not.
    np.save(
        tmp_path / pair_data_persistence.SELECTED_PAIR_INDICES_FILENAME,
        np.zeros((1, 2), dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Incomplete selected-pair schedule"):
        create_spectrum_pair_generator(
            spectra,
            settings,
            pair_data_folder=tmp_path,
        )
