"""Explicit persistence helpers for MS2DeepScore training-pair preparation.

This module intentionally does *not* implement an automatic cache. A user-provided
folder is treated as an explicit checkpoint location:

- an empty/new folder is populated during pair preparation;
- a populated folder is reused only when its manifest exactly matches the current
  structural inputs and pair-selection settings;
- a mismatch raises an error instead of silently reusing or overwriting data.

The expensive candidate pair/score matrices are stored in a compact CSR-like
representation per similarity bin. The final selected pair schedule is stored as
integer indices into a fixed-width InChIKey table plus float32 scores.
"""

from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np


PAIR_DATA_FORMAT_VERSION = 1
MANIFEST_FILENAME = "pair_data_manifest.json"
CANDIDATE_DATA_FILENAME = "candidate_pairs_and_scores.npz"
CANDIDATE_INCHIKEYS_FILENAME = "candidate_inchikeys.npy"
SELECTED_PAIR_INDICES_FILENAME = "selected_pair_indices.npy"
SELECTED_PAIR_SCORES_FILENAME = "selected_pair_scores.npy"
SELECTED_PAIR_INCHIKEYS_FILENAME = "selected_pair_inchikeys.npy"


@dataclass(frozen=True)
class SelectedPairSchedule:
    """Compact representation of an expanded selected-pair schedule.

    ``pair_indices`` contains two integer indices into ``inchikeys`` for each
    scheduled pair occurrence. Repeated selections remain repeated rows, matching
    the semantics of the historical list-of-tuples representation.
    """

    inchikeys: np.ndarray
    pair_indices: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        if self.pair_indices.ndim != 2 or self.pair_indices.shape[1] != 2:
            raise ValueError("pair_indices must have shape (n_pairs, 2).")
        if self.scores.ndim != 1:
            raise ValueError("scores must be a one-dimensional array.")
        if len(self.pair_indices) != len(self.scores):
            raise ValueError("pair_indices and scores must contain the same number of entries.")

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def __getitem__(self, index: int):
        pair = self.pair_indices[index]
        return (
            str(self.inchikeys[int(pair[0])]),
            str(self.inchikeys[int(pair[1])]),
            float(self.scores[index]),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[tuple[str, str, float]],
        inchikeys: Sequence[str] | None = None,
    ) -> "SelectedPairSchedule":
        if inchikeys is None:
            inchikeys = sorted({key for pair in pairs for key in pair[:2]})
        inchikey_array = np.asarray(inchikeys, dtype="U14")
        index_lookup = {key: i for i, key in enumerate(inchikey_array.tolist())}

        pair_indices = np.empty((len(pairs), 2), dtype=np.int32)
        scores = np.empty(len(pairs), dtype=np.float32)
        for i, (key_1, key_2, score) in enumerate(pairs):
            pair_indices[i, 0] = index_lookup[key_1]
            pair_indices[i, 1] = index_lookup[key_2]
            scores[i] = score

        return cls(inchikey_array, pair_indices, scores)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def spectra_structure_signature(spectra: Iterable) -> str:
    """Create an order-independent signature of structure metadata used for pairing.

    This deliberately hashes only the structural identifiers/annotations relevant to
    fingerprint-based pair selection. It is stricter than strictly necessary (for
    example, adding/removing a duplicate spectrum changes the signature), which is a
    safe choice for explicit checkpoint reuse.
    """
    records = []
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")
        records.append(
            (
                None if inchikey is None else inchikey[:14],
                spectrum.get("smiles"),
                spectrum.get("inchi"),
            )
        )
    records.sort(key=lambda x: tuple("" if v is None else str(v) for v in x))

    digest = sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_pair_data_manifest(
    *,
    kind: str,
    settings,
    spectra_signature_1: str,
    spectra_signature_2: str | None = None,
) -> dict:
    """Build the manifest describing exactly which persisted pair data is valid."""
    candidate_settings = {
        "fingerprint_type": settings.fingerprint_type,
        "fingerprint_nbits": settings.fingerprint_nbits,
        "max_pairs_per_bin": settings.max_pairs_per_bin,
        "same_prob_bins": np.asarray(settings.same_prob_bins).tolist(),
        "include_diagonal": settings.include_diagonal if kind == "within_set" else False,
        "random_seed": settings.random_seed,
    }
    balancing_settings = {
        "average_inchikey_sampling_count": settings.average_inchikey_sampling_count,
        "max_inchikey_sampling": settings.max_inchikey_sampling,
        "max_pair_resampling": settings.max_pair_resampling,
    }

    manifest = {
        "format_version": PAIR_DATA_FORMAT_VERSION,
        "kind": kind,
        "spectra_signature_1": spectra_signature_1,
        "candidate_settings": candidate_settings,
        "balancing_settings": balancing_settings,
    }
    if spectra_signature_2 is not None:
        manifest["spectra_signature_2"] = spectra_signature_2
    return _jsonable(manifest)


def _atomic_json_dump(data: dict, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as file:
        np.save(file, array, allow_pickle=False)
    os.replace(tmp_path, path)


def _atomic_save_npz(path: Path, **arrays) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as file:
        np.savez(file, **arrays)
    os.replace(tmp_path, path)


def prepare_pair_data_folder(folder: str | Path, expected_manifest: dict) -> Path:
    """Create/validate an explicit pair-data folder.

    A manifest mismatch raises a ValueError. Existing data is never silently
    overwritten or reused with different settings/input structures.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / MANIFEST_FILENAME

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as file:
            actual_manifest = json.load(file)
        if actual_manifest != expected_manifest:
            mismatches = []
            keys = sorted(set(actual_manifest) | set(expected_manifest))
            for key in keys:
                if actual_manifest.get(key) != expected_manifest.get(key):
                    mismatches.append(
                        f"  - {key}: stored={actual_manifest.get(key)!r}, current={expected_manifest.get(key)!r}"
                    )
            mismatch_text = "\n".join(mismatches)
            raise ValueError(
                "The supplied pair_data_folder contains pair data for different inputs/settings.\n"
                f"Folder: {folder}\n"
                f"Differences:\n{mismatch_text}\n"
                "Use a different/empty folder (or deliberately remove the old folder) for the new pair preparation."
            )
        return folder

    known_data_files = [
        CANDIDATE_DATA_FILENAME,
        CANDIDATE_INCHIKEYS_FILENAME,
        SELECTED_PAIR_INDICES_FILENAME,
        SELECTED_PAIR_SCORES_FILENAME,
        SELECTED_PAIR_INCHIKEYS_FILENAME,
    ]
    present_without_manifest = [name for name in known_data_files if (folder / name).exists()]
    if present_without_manifest:
        raise ValueError(
            "The supplied pair_data_folder contains pair-data files but no manifest, so it cannot be safely reused. "
            f"Folder: {folder}. Files: {present_without_manifest}"
        )

    _atomic_json_dump(expected_manifest, manifest_path)
    return folder


def selected_schedule_exists(folder: str | Path) -> bool:
    folder = Path(folder)
    paths = [
        folder / SELECTED_PAIR_INDICES_FILENAME,
        folder / SELECTED_PAIR_SCORES_FILENAME,
        folder / SELECTED_PAIR_INCHIKEYS_FILENAME,
    ]
    existing = [path.exists() for path in paths]
    if any(existing) and not all(existing):
        raise ValueError(f"Incomplete selected-pair schedule in {folder}.")
    return all(existing)


def candidate_data_exists(folder: str | Path) -> bool:
    folder = Path(folder)
    paths = [folder / CANDIDATE_DATA_FILENAME, folder / CANDIDATE_INCHIKEYS_FILENAME]
    existing = [path.exists() for path in paths]
    if any(existing) and not all(existing):
        raise ValueError(f"Incomplete candidate pair/score data in {folder}.")
    return all(existing)


def save_selected_pair_schedule(folder: str | Path, schedule: SelectedPairSchedule) -> None:
    """Persist the final selected schedule in compact, mmap-friendly arrays."""
    folder = Path(folder)
    _atomic_save_npy(folder / SELECTED_PAIR_INDICES_FILENAME, np.asarray(schedule.pair_indices, dtype=np.int32))
    _atomic_save_npy(folder / SELECTED_PAIR_SCORES_FILENAME, np.asarray(schedule.scores, dtype=np.float32))
    _atomic_save_npy(folder / SELECTED_PAIR_INCHIKEYS_FILENAME, np.asarray(schedule.inchikeys, dtype="S14"))


def load_selected_pair_schedule(folder: str | Path) -> SelectedPairSchedule:
    """Load the final selected schedule without materializing millions of Python tuples."""
    folder = Path(folder)
    if not selected_schedule_exists(folder):
        raise FileNotFoundError(f"No complete selected-pair schedule found in {folder}.")

    pair_indices = np.load(folder / SELECTED_PAIR_INDICES_FILENAME, mmap_mode="r", allow_pickle=False)
    scores = np.load(folder / SELECTED_PAIR_SCORES_FILENAME, mmap_mode="r", allow_pickle=False)
    inchikeys_bytes = np.load(folder / SELECTED_PAIR_INCHIKEYS_FILENAME, allow_pickle=False)
    inchikeys = inchikeys_bytes.astype("U14")
    return SelectedPairSchedule(inchikeys, pair_indices, scores)


def save_candidate_pair_data(
    folder: str | Path,
    available_pairs_per_bin_matrix: np.ndarray,
    available_scores_per_bin_matrix: np.ndarray,
    inchikeys14_unique: Sequence[str],
) -> None:
    """Persist candidate pairs/scores in a compact CSR-like representation per bin.

    The dense matrices use ``-1`` to indicate unused slots. We save only valid
    entries plus one ``indptr`` vector per bin. Rows are reconstructed left-packed,
    which preserves the semantics of the original candidate matrices.
    """
    if available_pairs_per_bin_matrix.shape != available_scores_per_bin_matrix.shape:
        raise ValueError("Candidate pair and score matrices must have identical shapes.")
    if available_pairs_per_bin_matrix.ndim != 3:
        raise ValueError("Candidate pair and score matrices must be 3-dimensional.")

    shape = tuple(int(x) for x in available_pairs_per_bin_matrix.shape)
    arrays = {"shape": np.asarray(shape, dtype=np.int64)}

    for bin_index in range(shape[0]):
        pair_bin = available_pairs_per_bin_matrix[bin_index]
        score_bin = available_scores_per_bin_matrix[bin_index]
        valid = pair_bin >= 0
        counts = valid.sum(axis=1, dtype=np.int64)
        indptr = np.empty(shape[1] + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])

        arrays[f"bin_{bin_index}_indptr"] = indptr
        arrays[f"bin_{bin_index}_pairs"] = pair_bin[valid].astype(np.int32, copy=False)
        arrays[f"bin_{bin_index}_scores"] = score_bin[valid].astype(np.float32, copy=False)

    folder = Path(folder)
    _atomic_save_npz(folder / CANDIDATE_DATA_FILENAME, **arrays)
    _atomic_save_npy(folder / CANDIDATE_INCHIKEYS_FILENAME, np.asarray(inchikeys14_unique, dtype="S14"))


def load_candidate_pair_data(folder: str | Path):
    """Load CSR-like candidate data and reconstruct the matrices needed by balancing."""
    folder = Path(folder)
    if not candidate_data_exists(folder):
        raise FileNotFoundError(f"No complete candidate pair/score data found in {folder}.")

    with np.load(folder / CANDIDATE_DATA_FILENAME, allow_pickle=False) as data:
        shape = tuple(int(x) for x in data["shape"])
        if len(shape) != 3:
            raise ValueError(f"Invalid candidate matrix shape stored in {folder}: {shape}")

        pair_matrix = np.full(shape, -1, dtype=np.int32)
        score_matrix = np.zeros(shape, dtype=np.float32)
        slot_numbers = np.arange(shape[2])[None, :]

        for bin_index in range(shape[0]):
            indptr = data[f"bin_{bin_index}_indptr"]
            pairs = data[f"bin_{bin_index}_pairs"]
            scores = data[f"bin_{bin_index}_scores"]
            counts = np.diff(indptr)
            if int(counts.sum()) != len(pairs) or len(pairs) != len(scores):
                raise ValueError(f"Corrupt candidate data for bin {bin_index} in {folder}.")
            if np.any(counts > shape[2]):
                raise ValueError(f"Corrupt candidate row length in bin {bin_index} in {folder}.")

            valid = slot_numbers < counts[:, None]
            pair_matrix[bin_index][valid] = pairs
            score_matrix[bin_index][valid] = scores

    inchikeys_bytes = np.load(folder / CANDIDATE_INCHIKEYS_FILENAME, allow_pickle=False)
    inchikeys = inchikeys_bytes.astype("U14").tolist()
    if len(inchikeys) != shape[1]:
        raise ValueError(
            f"Stored candidate InChIKey count ({len(inchikeys)}) does not match candidate matrix rows ({shape[1]})."
        )
    return pair_matrix, score_matrix, inchikeys
