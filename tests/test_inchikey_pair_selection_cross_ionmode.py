import numpy as np
import pytest
from matchms import Spectrum

from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.inchikey_pair_selection_cross_ionmode import (
    create_data_generator_across_ionmodes,
    select_compound_pairs_wrapper_across_ionmode,
    SpectrumPairGeneratorAcrossIonmodes,
    CombinedSpectrumGenerator,
)
from ms2deepscore.train_new_model import SpectrumPairGenerator
from tests.create_test_spectra import create_test_spectra


def _make_cross_ionmode_settings(
    fingerprint_type="rdkit_binary",
    bins=None,
    batch_size=2,
    average_inchikey_sampling_count=4,
    max_pair_resampling=10,
    max_inchikey_sampling=100,
):
    if bins is None:
        bins = [(-0.01, 1.0)]
    return SettingsMS2Deepscore(
        min_mz=10,
        max_mz=1000,
        mz_bin_width=0.1,
        intensity_scaling=0.5,
        additional_metadata=[],
        same_prob_bins=np.array(bins, dtype="float32"),
        batch_size=batch_size,
        num_turns=4,
        average_inchikey_sampling_count=average_inchikey_sampling_count,
        max_pair_resampling=max_pair_resampling,
        max_inchikey_sampling=max_inchikey_sampling,
        fingerprint_type=fingerprint_type,
        fingerprint_nbits=256,
        augment_removal_max=0.0,
        augment_removal_intensity=0.0,
        augment_intensity=0.0,
        augment_noise_max=0.0,
    )


@pytest.fixture
def pos_neg_spectra():
    test_spectra = create_test_spectra(20, 2)

    pos_spectra = []
    for spectrum in test_spectra[:20]:
        spectrum.set("ionmode", "positive")
        pos_spectra.append(spectrum)

    neg_spectra = []
    for spectrum in test_spectra[20:]:
        spectrum.set("ionmode", "negative")
        neg_spectra.append(spectrum)

    return pos_spectra, neg_spectra


@pytest.fixture
def small_pos_neg_spectra():
    pos_spectra = [
        Spectrum(
            mz=np.array([100.0]),
            intensities=np.array([1.0]),
            metadata={"inchikey": "AAAAAAAAAAAAAA-AAAAAAAAAA-N", "ionmode": "positive"},
        ),
        Spectrum(
            mz=np.array([110.0]),
            intensities=np.array([0.8]),
            metadata={"inchikey": "BBBBBBBBBBBBBB-BBBBBBBBBB-N", "ionmode": "positive"},
        ),
    ]
    neg_spectra = [
        Spectrum(
            mz=np.array([120.0]),
            intensities=np.array([0.7]),
            metadata={"inchikey": "CCCCCCCCCCCCCC-CCCCCCCCCC-N", "ionmode": "negative"},
        ),
        Spectrum(
            mz=np.array([130.0]),
            intensities=np.array([0.6]),
            metadata={"inchikey": "DDDDDDDDDDDDDD-DDDDDDDDDD-N", "ionmode": "negative"},
        ),
    ]
    return pos_spectra, neg_spectra


@pytest.mark.parametrize(
    "fingerprint_type,bins",
    [
        ("rdkit_binary", [(-0.01, 0.6), (0.6, 1.0)]),
        ("rdkit_count", [(-0.01, 0.6), (0.6, 1.0)]),
        ("rdkit_binary_unfolded", [(-0.01, 0.6), (0.6, 1.0)]),
        ("rdkit_count_unfolded", [(-0.01, 0.6), (0.6, 1.0)]),
    ],
)
def test_select_compound_pairs_wrapper_across_ionmode_supported_types(
    pos_neg_spectra, fingerprint_type, bins
):
    pos_spectra, neg_spectra = pos_neg_spectra
    settings = _make_cross_ionmode_settings(
        fingerprint_type=fingerprint_type,
        bins=bins,
        batch_size=2,
        average_inchikey_sampling_count=4,
    )

    spectrum_pair_generator = select_compound_pairs_wrapper_across_ionmode(
        pos_spectra, neg_spectra, settings
    )

    assert len(spectrum_pair_generator) > 0

    for _ in range(len(spectrum_pair_generator)):
        spectrum_1, spectrum_2, score = next(spectrum_pair_generator)
        assert spectrum_1.get("ionmode") == "positive"
        assert spectrum_2.get("ionmode") == "negative"
        assert 0.0 <= score <= 1.0

    # Infinite generator behavior
    spectrum_1, spectrum_2, score = next(spectrum_pair_generator)
    assert spectrum_1.get("ionmode") == "positive"
    assert spectrum_2.get("ionmode") == "negative"
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    "fingerprint_type,bins",
    [
        ("rdkit_binary", [(-0.01, 1.0)]),
        ("rdkit_count", [(-0.01, 1.0)]),
        ("rdkit_binary_unfolded", [(-0.01, 1.0)]),
        ("rdkit_count_unfolded", [(-0.01, 1.0)]),
    ],
)
def test_create_data_generator_across_ionmodes_supported_types(
    pos_neg_spectra, fingerprint_type, bins
):
    pos_spectra, neg_spectra = pos_neg_spectra
    settings = _make_cross_ionmode_settings(
        fingerprint_type=fingerprint_type,
        bins=bins,
        batch_size=2,
        average_inchikey_sampling_count=4,
    )

    data_generator = create_data_generator_across_ionmodes(pos_spectra + neg_spectra, settings)

    assert len(data_generator) > 0

    for _ in range(len(data_generator)):
        spec1, spec2, meta1, meta2, targets = next(data_generator)
        assert spec1.shape[0] == settings.batch_size
        assert spec2.shape[0] == settings.batch_size
        assert meta1.shape[0] == settings.batch_size
        assert meta2.shape[0] == settings.batch_size
        assert targets.shape[0] == settings.batch_size


def test_spectrum_pair_generator_across_ionmodes_get_scores_and_counts(small_pos_neg_spectra):
    pos_spectra, neg_spectra = small_pos_neg_spectra
    selected_inchikey_pairs = [
        ("AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC", 0.2),
        ("AAAAAAAAAAAAAA", "DDDDDDDDDDDDDD", 0.6),
        ("BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", 0.8),
    ]

    generator = SpectrumPairGeneratorAcrossIonmodes(
        selected_inchikey_pairs=selected_inchikey_pairs,
        spectra_pos=pos_spectra,
        spectra_neg=neg_spectra,
        shuffle=False,
        random_seed=0,
    )

    assert generator.get_scores() == [0.2, 0.6, 0.8]

    counts = generator.get_inchikey_counts()
    assert counts["AAAAAAAAAAAAAA"] == 2
    assert counts["BBBBBBBBBBBBBB"] == 1
    assert counts["CCCCCCCCCCCCCC"] == 2
    assert counts["DDDDDDDDDDDDDD"] == 1


def test_spectrum_pair_generator_across_ionmodes_get_scores_per_inchikey(small_pos_neg_spectra):
    pos_spectra, neg_spectra = small_pos_neg_spectra
    selected_inchikey_pairs = [
        ("AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC", 0.2),
        ("AAAAAAAAAAAAAA", "DDDDDDDDDDDDDD", 0.6),
        ("BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", 0.8),
    ]

    generator = SpectrumPairGeneratorAcrossIonmodes(
        selected_inchikey_pairs=selected_inchikey_pairs,
        spectra_pos=pos_spectra,
        spectra_neg=neg_spectra,
        shuffle=False,
        random_seed=0,
    )

    scores_per_inchikey = generator.get_scores_per_inchikey()

    assert scores_per_inchikey["AAAAAAAAAAAAAA"] == [0.2, 0.6]
    assert scores_per_inchikey["BBBBBBBBBBBBBB"] == [0.8]
    assert scores_per_inchikey["CCCCCCCCCCCCCC"] == [0.2, 0.8]
    assert scores_per_inchikey["DDDDDDDDDDDDDD"] == [0.6]


def test_spectrum_pair_generator_across_ionmodes_without_shuffle_order(small_pos_neg_spectra):
    pos_spectra, neg_spectra = small_pos_neg_spectra
    selected_inchikey_pairs = [
        ("AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC", 0.2),
        ("BBBBBBBBBBBBBB", "DDDDDDDDDDDDDD", 0.6),
    ]

    generator = SpectrumPairGeneratorAcrossIonmodes(
        selected_inchikey_pairs=selected_inchikey_pairs,
        spectra_pos=pos_spectra,
        spectra_neg=neg_spectra,
        shuffle=False,
        random_seed=0,
    )

    spectrum_1, spectrum_2, score = next(generator)
    assert spectrum_1.get("inchikey")[:14] == "AAAAAAAAAAAAAA"
    assert spectrum_2.get("inchikey")[:14] == "CCCCCCCCCCCCCC"
    assert score == 0.2

    spectrum_1, spectrum_2, score = next(generator)
    assert spectrum_1.get("inchikey")[:14] == "BBBBBBBBBBBBBB"
    assert spectrum_2.get("inchikey")[:14] == "DDDDDDDDDDDDDD"
    assert score == 0.6


def test_spectrum_pair_generator_across_ionmodes_missing_positive_inchikey_raises(small_pos_neg_spectra):
    pos_spectra, neg_spectra = small_pos_neg_spectra
    generator = SpectrumPairGeneratorAcrossIonmodes(
        selected_inchikey_pairs=[("ZZZZZZZZZZZZZZ", "CCCCCCCCCCCCCC", 0.2)],
        spectra_pos=pos_spectra,
        spectra_neg=neg_spectra,
        shuffle=False,
        random_seed=0,
    )

    with pytest.raises(ValueError, match="No matching positive-mode inchikey found"):
        next(generator)


def test_spectrum_pair_generator_across_ionmodes_missing_negative_inchikey_raises(small_pos_neg_spectra):
    pos_spectra, neg_spectra = small_pos_neg_spectra
    generator = SpectrumPairGeneratorAcrossIonmodes(
        selected_inchikey_pairs=[("AAAAAAAAAAAAAA", "ZZZZZZZZZZZZZZ", 0.2)],
        spectra_pos=pos_spectra,
        spectra_neg=neg_spectra,
        shuffle=False,
        random_seed=0,
    )

    with pytest.raises(ValueError, match="No matching negative-mode inchikey found"):
        next(generator)


def test_combined_spectrum_generator_cycles_through_generators():
    gen1 = SpectrumPairGenerator(
        [("A", "B", 0.1)],
        [
            Spectrum(mz=np.array([100.0]), intensities=np.array([1.0]), metadata={"inchikey": "A"}),
            Spectrum(mz=np.array([101.0]), intensities=np.array([1.0]), metadata={"inchikey": "B"}),
        ],
        shuffle=False,
        random_seed=0,
    )
    gen2 = SpectrumPairGenerator(
        [("C", "D", 0.2)],
        [
            Spectrum(mz=np.array([102.0]), intensities=np.array([1.0]), metadata={"inchikey": "C"}),
            Spectrum(mz=np.array([103.0]), intensities=np.array([1.0]), metadata={"inchikey": "D"}),
        ],
        shuffle=False,
        random_seed=0,
    )

    combined = CombinedSpectrumGenerator([gen1, gen2])

    spectrum_1, spectrum_2, score = next(combined)
    assert score == 0.1

    spectrum_1, spectrum_2, score = next(combined)
    assert score == 0.2

    spectrum_1, spectrum_2, score = next(combined)
    assert score == 0.1

    assert len(combined) == len(gen1) + len(gen2)


# -----------------------------------------------------------------------------
# Tests for explicit pair-data folders in cross-ionmode training
# -----------------------------------------------------------------------------

from collections import Counter

import ms2deepscore.train_new_model.inchikey_pair_selection_cross_ionmode as cross_pair_selection_module
import ms2deepscore.train_new_model.pair_data_persistence as pair_data_persistence


def test_create_data_generator_across_ionmodes_uses_explicit_subfolders(
    tmp_path, monkeypatch, pos_neg_spectra
):
    """The top-level folder is only a layout; each pairing mode gets an explicit subfolder."""
    pos_spectra, neg_spectra = pos_neg_spectra
    settings = _make_cross_ionmode_settings()
    recorded_folders = []

    class DummyPairGenerator:
        def __len__(self):
            return 1

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    def fake_within(spectra, settings, pair_data_folder=None):
        recorded_folders.append(pair_data_folder)
        return DummyPairGenerator()

    def fake_cross(spectra_1, spectra_2, settings, pair_data_folder=None):
        recorded_folders.append(pair_data_folder)
        return DummyPairGenerator()

    monkeypatch.setattr(cross_pair_selection_module, "create_spectrum_pair_generator", fake_within)
    monkeypatch.setattr(
        cross_pair_selection_module,
        "select_compound_pairs_wrapper_across_ionmode",
        fake_cross,
    )
    # We only test folder plumbing here; constructing batches is covered elsewhere.
    monkeypatch.setattr(
        cross_pair_selection_module,
        "TrainingBatchGenerator",
        lambda spectrum_pair_generator, settings: spectrum_pair_generator,
    )

    create_data_generator_across_ionmodes(
        pos_spectra + neg_spectra,
        settings,
        pair_data_folder=tmp_path,
    )

    assert recorded_folders == [
        tmp_path / "positive",
        tmp_path / "negative",
        tmp_path / "positive_negative",
    ]
    assert (tmp_path / pair_data_persistence.MANIFEST_FILENAME).is_file()


def _make_cross_pair_folder_settings():
    settings = _make_cross_ionmode_settings(
        bins=[(-0.01, 1.0)],
        batch_size=2,
        average_inchikey_sampling_count=2,
    )
    settings.max_pairs_per_bin = 2
    settings.random_seed = 17
    settings.shuffle = False
    return settings


def _install_fake_cross_pair_preparation(
    monkeypatch,
    pos_spectra,
    neg_spectra,
    selected_pairs,
    calls,
):
    pos_keys = sorted({s.get("inchikey")[:14] for s in pos_spectra})
    neg_keys = sorted({s.get("inchikey")[:14] for s in neg_spectra})
    all_keys = pos_keys + neg_keys

    def fake_fingerprints(spectra, *args, **kwargs):
        calls["fingerprints"] += 1
        keys = sorted({s.get("inchikey")[:14] for s in spectra})
        return np.zeros((len(keys), 8), dtype=np.float32), keys

    candidate_pairs = np.full((1, len(all_keys), 1), -1, dtype=np.int32)
    candidate_scores = np.zeros_like(candidate_pairs, dtype=np.float32)
    # The actual contents are irrelevant here because balancing/conversion are mocked,
    # but use a valid global target index in one row so persistence is non-empty.
    candidate_pairs[0, 0, 0] = len(pos_keys)
    candidate_scores[0, 0, 0] = 0.5

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
        cross_pair_selection_module,
        "compute_fingerprints_for_training",
        fake_fingerprints,
    )
    monkeypatch.setattr(
        cross_pair_selection_module,
        "compute_tanimoto_similarity_per_bin_between_sets",
        fake_candidates,
    )
    monkeypatch.setattr(
        cross_pair_selection_module,
        "balanced_selection_of_pairs_per_bin",
        fake_balancing,
    )
    monkeypatch.setattr(
        cross_pair_selection_module,
        "convert_to_selected_pairs_list",
        fake_conversion,
    )


def test_cross_ionmode_pair_folder_second_run_reuses_final_schedule(
    tmp_path, monkeypatch, pos_neg_spectra
):
    pos_spectra, neg_spectra = pos_neg_spectra
    settings = _make_cross_pair_folder_settings()
    pos_keys = sorted({s.get("inchikey")[:14] for s in pos_spectra})
    neg_keys = sorted({s.get("inchikey")[:14] for s in neg_spectra})
    selected_pairs = [
        (pos_keys[0], neg_keys[0], 0.2),
        (pos_keys[1], neg_keys[1], 0.8),
    ]
    calls = Counter()
    _install_fake_cross_pair_preparation(
        monkeypatch,
        pos_spectra,
        neg_spectra,
        selected_pairs,
        calls,
    )

    first = select_compound_pairs_wrapper_across_ionmode(
        pos_spectra,
        neg_spectra,
        settings,
        pair_data_folder=tmp_path,
    )
    first_pairs = list(first.selected_inchikey_pairs)

    def unexpected_call(*args, **kwargs):
        pytest.fail("Cross-ionmode pair preparation should not run when a final schedule exists.")

    monkeypatch.setattr(cross_pair_selection_module, "compute_fingerprints_for_training", unexpected_call)
    monkeypatch.setattr(
        cross_pair_selection_module,
        "compute_tanimoto_similarity_per_bin_between_sets",
        unexpected_call,
    )
    monkeypatch.setattr(cross_pair_selection_module, "balanced_selection_of_pairs_per_bin", unexpected_call)
    monkeypatch.setattr(cross_pair_selection_module, "convert_to_selected_pairs_list", unexpected_call)

    second = select_compound_pairs_wrapper_across_ionmode(
        pos_spectra,
        neg_spectra,
        settings,
        pair_data_folder=tmp_path,
    )
    second_pairs = list(second.selected_inchikey_pairs)

    assert [pair[:2] for pair in second_pairs] == [pair[:2] for pair in first_pairs]
    np.testing.assert_allclose(
        [pair[2] for pair in second_pairs],
        [pair[2] for pair in first_pairs],
    )
    assert pair_data_persistence.candidate_data_exists(tmp_path)
    assert pair_data_persistence.selected_schedule_exists(tmp_path)


def test_cross_ionmode_pair_folder_reuses_candidate_checkpoint(
    tmp_path, monkeypatch, pos_neg_spectra
):
    pos_spectra, neg_spectra = pos_neg_spectra
    settings = _make_cross_pair_folder_settings()
    pos_keys = sorted({s.get("inchikey")[:14] for s in pos_spectra})
    neg_keys = sorted({s.get("inchikey")[:14] for s in neg_spectra})
    all_keys = pos_keys + neg_keys

    manifest = pair_data_persistence.build_pair_data_manifest(
        kind="between_sets",
        settings=settings,
        spectra_signature_1=pair_data_persistence.spectra_structure_signature(pos_spectra),
        spectra_signature_2=pair_data_persistence.spectra_structure_signature(neg_spectra),
    )
    pair_data_persistence.prepare_pair_data_folder(tmp_path, manifest)

    candidate_pairs = np.full((1, len(all_keys), 1), -1, dtype=np.int32)
    candidate_scores = np.zeros_like(candidate_pairs, dtype=np.float32)
    candidate_pairs[0, 0, 0] = len(pos_keys)
    candidate_scores[0, 0, 0] = 0.5
    pair_data_persistence.save_candidate_pair_data(
        tmp_path,
        candidate_pairs,
        candidate_scores,
        all_keys,
    )

    selected_pairs = [(pos_keys[0], neg_keys[0], 0.5)]
    calls = Counter()

    def unexpected_call(*args, **kwargs):
        pytest.fail("Fingerprint/candidate computation should be skipped when cross candidate data exists.")

    def fake_balancing(*args, **kwargs):
        calls["balancing"] += 1
        return np.zeros_like(candidate_pairs, dtype=np.int32)

    def fake_conversion(*args, **kwargs):
        calls["conversion"] += 1
        return [selected_pairs]

    monkeypatch.setattr(cross_pair_selection_module, "compute_fingerprints_for_training", unexpected_call)
    monkeypatch.setattr(
        cross_pair_selection_module,
        "compute_tanimoto_similarity_per_bin_between_sets",
        unexpected_call,
    )
    monkeypatch.setattr(cross_pair_selection_module, "balanced_selection_of_pairs_per_bin", fake_balancing)
    monkeypatch.setattr(cross_pair_selection_module, "convert_to_selected_pairs_list", fake_conversion)

    generator = select_compound_pairs_wrapper_across_ionmode(
        pos_spectra,
        neg_spectra,
        settings,
        pair_data_folder=tmp_path,
    )

    assert calls == Counter(balancing=1, conversion=1)
    assert pair_data_persistence.selected_schedule_exists(tmp_path)
    assert len(generator) == 1


def test_cross_ionmode_top_level_pair_folder_rejects_changed_pair_settings(
    tmp_path, monkeypatch, pos_neg_spectra
):
    pos_spectra, neg_spectra = pos_neg_spectra
    settings = _make_cross_pair_folder_settings()

    class DummyPairGenerator:
        def __len__(self):
            return 1

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    monkeypatch.setattr(
        cross_pair_selection_module,
        "create_spectrum_pair_generator",
        lambda *args, **kwargs: DummyPairGenerator(),
    )
    monkeypatch.setattr(
        cross_pair_selection_module,
        "select_compound_pairs_wrapper_across_ionmode",
        lambda *args, **kwargs: DummyPairGenerator(),
    )
    monkeypatch.setattr(
        cross_pair_selection_module,
        "TrainingBatchGenerator",
        lambda spectrum_pair_generator, settings: spectrum_pair_generator,
    )

    create_data_generator_across_ionmodes(
        pos_spectra + neg_spectra,
        settings,
        pair_data_folder=tmp_path,
    )

    changed_settings = _make_cross_pair_folder_settings()
    changed_settings.random_seed += 1

    with pytest.raises(ValueError, match="different inputs/settings"):
        create_data_generator_across_ionmodes(
            pos_spectra + neg_spectra,
            changed_settings,
            pair_data_folder=tmp_path,
        )
