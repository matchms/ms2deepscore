"""Persistent caches for expensive training-pair preparation.

Pair selection naturally separates into two reusable artifacts:

1. Candidate pairs per Tanimoto bin (expensive all-vs-all fingerprint work).
2. The final balanced pair schedule (depends on balancing settings as well).

Both artifacts are keyed by the structural metadata of the input spectra and by
only the settings that can affect the corresponding artifact. Arrays are stored
as .npy files so large candidate pools can be memory-mapped on reload.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np


CACHE_SCHEMA_VERSION = 1
CANDIDATE_ALGORITHM_VERSION = 1
SELECTION_ALGORITHM_VERSION = 1


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _serialize_bins(score_bins) -> list[list[float]]:
    return [[float(low), float(high)] for low, high in np.asarray(score_bins)]


def _spectrum_metadata_signature(spectra_sets: Sequence[Sequence]) -> str:
    """Hash only metadata that can affect structure-based pair selection.

    Peak arrays deliberately are not hashed because they do not participate in
    fingerprint/Tanimoto pair selection. Input order is included because legacy
    representative-structure selection can use the first matching spectrum.
    """
    digest = hashlib.sha256()
    for set_index, spectra in enumerate(spectra_sets):
        digest.update(f"SET:{set_index}\n".encode("ascii"))
        for spectrum in spectra:
            if spectrum is None:
                payload = [None, None, None, None]
            else:
                payload = [
                    spectrum.get("inchikey"),
                    spectrum.get("smiles"),
                    spectrum.get("inchi"),
                    spectrum.get("ionmode"),
                ]
            digest.update(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def _candidate_parameters(settings, mode: str, dataset_signature: str) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "algorithm_version": CANDIDATE_ALGORITHM_VERSION,
        "mode": mode,
        "dataset_signature": dataset_signature,
        "fingerprint_type": settings.fingerprint_type,
        "fingerprint_nbits": int(settings.fingerprint_nbits),
        "max_pairs_per_bin": None
        if settings.max_pairs_per_bin is None
        else int(settings.max_pairs_per_bin),
        "same_prob_bins": _serialize_bins(settings.same_prob_bins),
        "include_diagonal": bool(settings.include_diagonal),
        # Candidate subsampling uses random shuffling when a bin contains more
        # than max_pairs_per_bin entries, so seed is part of artifact identity.
        "random_seed": settings.random_seed,
    }


def _selection_parameters(settings) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "algorithm_version": SELECTION_ALGORITHM_VERSION,
        "average_inchikey_sampling_count": float(settings.average_inchikey_sampling_count),
        "max_inchikey_sampling": int(settings.max_inchikey_sampling),
        "max_pair_resampling": int(settings.max_pair_resampling),
    }


class PairSelectionCache:
    """Read/write persistent pair-selection artifacts under one cache root."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _candidate_dir(self, spectra_sets, settings, mode: str):
        dataset_signature = _spectrum_metadata_signature(spectra_sets)
        params = _candidate_parameters(settings, mode, dataset_signature)
        return self.root / f"candidates_{_json_hash(params)}", params

    @staticmethod
    def _metadata_matches(path: Path, expected: dict) -> bool:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle) == expected
        except (OSError, json.JSONDecodeError):
            return False

    def load_candidates(self, spectra_sets, settings, mode: str):
        candidate_dir, params = self._candidate_dir(spectra_sets, settings, mode)
        metadata_path = candidate_dir / "metadata.json"
        required = (
            candidate_dir / "inchikeys.npy",
            candidate_dir / "available_pairs.npy",
            candidate_dir / "available_scores.npy",
        )
        if not self._metadata_matches(metadata_path, params) or not all(path.exists() for path in required):
            return None

        inchikeys = np.load(required[0], allow_pickle=False).astype("U14").tolist()
        available_pairs = np.load(required[1], mmap_mode="r", allow_pickle=False)
        available_scores = np.load(required[2], mmap_mode="r", allow_pickle=False)
        return inchikeys, available_pairs, available_scores, candidate_dir

    def save_candidates(
        self,
        spectra_sets,
        settings,
        mode: str,
        inchikeys,
        available_pairs: np.ndarray,
        available_scores: np.ndarray,
    ) -> Path:
        candidate_dir, params = self._candidate_dir(spectra_sets, settings, mode)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        # InChIKey14 is ASCII. Store fixed-width bytes instead of NumPy Unicode
        # (14 vs. 56 bytes per key) to keep large cached pair schedules compact.
        np.save(candidate_dir / "inchikeys.npy", np.asarray(inchikeys, dtype="S14"), allow_pickle=False)
        np.save(
            candidate_dir / "available_pairs.npy",
            np.asarray(available_pairs, dtype=np.int32),
            allow_pickle=False,
        )
        np.save(
            candidate_dir / "available_scores.npy",
            np.asarray(available_scores, dtype=np.float32),
            allow_pickle=False,
        )

        tmp_metadata = candidate_dir / "metadata.json.tmp"
        with tmp_metadata.open("w", encoding="utf-8") as handle:
            json.dump(params, handle, indent=2, sort_keys=True)
        tmp_metadata.replace(candidate_dir / "metadata.json")
        return candidate_dir

    def _selection_dir(self, candidate_dir: Path, settings) -> tuple[Path, dict]:
        params = _selection_parameters(settings)
        return candidate_dir / f"selection_{_json_hash(params)}", params

    def load_selected_pairs(self, candidate_dir: Path, settings):
        selection_dir, params = self._selection_dir(candidate_dir, settings)
        metadata_path = selection_dir / "metadata.json"
        pair1_path = selection_dir / "inchikey_1.npy"
        pair2_path = selection_dir / "inchikey_2.npy"
        scores_path = selection_dir / "scores.npy"
        if not self._metadata_matches(metadata_path, params):
            return None
        if not (pair1_path.exists() and pair2_path.exists() and scores_path.exists()):
            return None

        pair_1 = np.load(pair1_path, mmap_mode="r", allow_pickle=False)
        pair_2 = np.load(pair2_path, mmap_mode="r", allow_pickle=False)
        scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
        if not (len(pair_1) == len(pair_2) == len(scores)):
            return None

        # SpectrumPairGenerator's public API currently consumes tuples. Keeping
        # that interface avoids a broad user-facing change while still skipping
        # all expensive pair-selection work on cache hits.
        return [
            (
                bytes(inchikey_1).decode("ascii"),
                bytes(inchikey_2).decode("ascii"),
                float(score),
            )
            for inchikey_1, inchikey_2, score in zip(pair_1, pair_2, scores)
        ]

    def save_selected_pairs(self, candidate_dir: Path, settings, selected_pairs) -> Path:
        selection_dir, params = self._selection_dir(candidate_dir, settings)
        selection_dir.mkdir(parents=True, exist_ok=True)

        n_pairs = len(selected_pairs)
        inchikey_1 = np.empty(n_pairs, dtype="S14")
        inchikey_2 = np.empty(n_pairs, dtype="S14")
        scores = np.empty(n_pairs, dtype=np.float32)
        for idx, (key_1, key_2, score) in enumerate(selected_pairs):
            inchikey_1[idx] = key_1
            inchikey_2[idx] = key_2
            scores[idx] = score

        np.save(selection_dir / "inchikey_1.npy", inchikey_1, allow_pickle=False)
        np.save(selection_dir / "inchikey_2.npy", inchikey_2, allow_pickle=False)
        np.save(selection_dir / "scores.npy", scores, allow_pickle=False)

        tmp_metadata = selection_dir / "metadata.json.tmp"
        with tmp_metadata.open("w", encoding="utf-8") as handle:
            json.dump(params, handle, indent=2, sort_keys=True)
        tmp_metadata.replace(selection_dir / "metadata.json")
        return selection_dir


def resolve_pair_selection_cache_directory(settings, fallback_root=None):
    """Return the shared cache directory requested by training settings."""
    if not getattr(settings, "use_pair_selection_cache", True):
        return None
    explicit = getattr(settings, "pair_selection_cache_directory", None)
    if explicit is not None:
        return explicit
    results_folder = getattr(settings, "results_folder", None)
    if results_folder is not None:
        return os.path.join(results_folder, "pair_selection_cache")
    if fallback_root is not None:
        return os.path.join(fallback_root, "pair_selection_cache")
    return None
