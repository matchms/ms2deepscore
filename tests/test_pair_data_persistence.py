import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "ms2deepscore"
    / "train_new_model"
    / "pair_data_persistence.py"
)
spec = importlib.util.spec_from_file_location("pair_data_persistence", MODULE_PATH)
persistence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(persistence)


class DummySettings:
    fingerprint_type = "rdkit_binary"
    fingerprint_nbits = 2048
    max_pairs_per_bin = 3
    same_prob_bins = np.array([[-0.01, 0.5], [0.5, 1.0]])
    include_diagonal = True
    random_seed = 42
    average_inchikey_sampling_count = 10
    max_inchikey_sampling = 12
    max_pair_resampling = 100


class DummySpectrum:
    def __init__(self, inchikey, smiles=None, inchi=None):
        self.values = {"inchikey": inchikey, "smiles": smiles, "inchi": inchi}

    def get(self, key):
        return self.values.get(key)


def test_candidate_pair_data_roundtrip(tmp_path):
    pairs = np.array(
        [
            [[1, 2, -1], [0, -1, -1], [0, -1, -1]],
            [[2, -1, -1], [2, 0, -1], [0, 1, -1]],
        ],
        dtype=np.int32,
    )
    scores = np.array(
        [
            [[0.2, 0.3, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
            [[0.8, 0.0, 0.0], [0.7, 0.6, 0.0], [0.8, 0.7, 0.0]],
        ],
        dtype=np.float32,
    )
    inchikeys = ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"]

    persistence.save_candidate_pair_data(tmp_path, pairs, scores, inchikeys)
    loaded_pairs, loaded_scores, loaded_inchikeys = persistence.load_candidate_pair_data(tmp_path)

    np.testing.assert_array_equal(loaded_pairs, pairs)
    np.testing.assert_allclose(loaded_scores, scores)
    assert loaded_inchikeys == inchikeys


def test_selected_pair_schedule_roundtrip(tmp_path):
    pairs = [
        ("AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", 0.25),
        ("AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", 0.25),
        ("BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", 0.75),
    ]
    schedule = persistence.SelectedPairSchedule.from_pairs(pairs)

    persistence.save_selected_pair_schedule(tmp_path, schedule)
    loaded = persistence.load_selected_pair_schedule(tmp_path)

    loaded_pairs = list(loaded)
    assert [pair[:2] for pair in loaded_pairs] == [pair[:2] for pair in pairs]
    np.testing.assert_allclose(
        [pair[2] for pair in loaded_pairs],
        [pair[2] for pair in pairs],
    )
    assert isinstance(loaded.pair_indices, np.memmap)
    assert isinstance(loaded.scores, np.memmap)


def test_manifest_rejects_changed_pair_selection_settings(tmp_path):
    spectra = [
        DummySpectrum("AAAAAAAAAAAAAA-XYZ", smiles="CC"),
        DummySpectrum("BBBBBBBBBBBBBB-XYZ", smiles="CCC"),
    ]
    signature = persistence.spectra_structure_signature(spectra)
    expected = persistence.build_pair_data_manifest(
        kind="within_set",
        settings=DummySettings(),
        spectra_signature_1=signature,
    )
    persistence.prepare_pair_data_folder(tmp_path, expected)

    changed = dict(expected)
    changed["candidate_settings"] = dict(expected["candidate_settings"])
    changed["candidate_settings"]["fingerprint_nbits"] = 4096

    with pytest.raises(ValueError, match="different inputs/settings"):
        persistence.prepare_pair_data_folder(tmp_path, changed)


def test_structure_signature_is_order_independent_but_data_sensitive():
    spectra_a = [
        DummySpectrum("AAAAAAAAAAAAAA-XYZ", smiles="CC"),
        DummySpectrum("BBBBBBBBBBBBBB-XYZ", smiles="CCC"),
    ]
    spectra_b = list(reversed(spectra_a))
    spectra_changed = [
        DummySpectrum("AAAAAAAAAAAAAA-XYZ", smiles="CC"),
        DummySpectrum("BBBBBBBBBBBBBB-XYZ", smiles="CCCC"),
    ]

    assert persistence.spectra_structure_signature(spectra_a) == persistence.spectra_structure_signature(spectra_b)
    assert persistence.spectra_structure_signature(spectra_a) != persistence.spectra_structure_signature(spectra_changed)
