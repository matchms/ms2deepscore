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

    with pytest.raises(ValueError, match="No matching inchikey found"):
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

    with pytest.raises(ValueError, match="No matching inchikey found"):
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
