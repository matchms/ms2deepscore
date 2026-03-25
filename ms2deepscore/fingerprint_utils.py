from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from chemap import compute_fingerprints, FingerprintConfig


def _inchi_to_smiles(inchi: str) -> str | None:
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def normalize_to_smiles(smiles_or_inchi: str | List[str]) -> str | List[str | None]:
    """
    Convert InChI entries to SMILES. Leave SMILES unchanged.
    Invalid InChI entries return None.
    """
    if isinstance(smiles_or_inchi, str):
        if smiles_or_inchi.startswith("InChI="):
            return _inchi_to_smiles(smiles_or_inchi)
        return smiles_or_inchi

    normalized = []
    for entry in smiles_or_inchi:
        if entry is None:
            normalized.append(None)
        elif entry.startswith("InChI="):
            normalized.append(_inchi_to_smiles(entry))
        else:
            normalized.append(entry)
    return normalized


def derive_fingerprint_from_smiles(
        smiles: str | List[str],
        fingerprint_type="rdkit_binary",
        nbits=2048,
        policy_invalid_smiles="raise",
        ) -> np.ndarray:
    if fingerprint_type not in {"rdkit_binary", "rdkit_count"}:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")

    generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nbits)

    is_single = isinstance(smiles, str)
    inputs = [smiles] if is_single else smiles

    fingerprints = compute_fingerprints(
        inputs,
        generator,
        config=FingerprintConfig(
            count=(fingerprint_type == "rdkit_count"),
            folded=True,
            return_csr=False,
            invalid_policy=policy_invalid_smiles,
        ),
    )

    if not isinstance(fingerprints, np.ndarray):
        raise ValueError("Fingerprint computation failed.")

    return fingerprints[0] if is_single else fingerprints


def derive_fingerprint_from_smiles_or_inchi(
        smiles_or_inchi: str | List[str],
        fingerprint_type="rdkit_binary",
        nbits=2048,
        policy_invalid="raise",
        ) -> np.ndarray:
    normalized = normalize_to_smiles(smiles_or_inchi)

    if normalized is None:
        if policy_invalid == "raise":
            raise ValueError("Could not convert input structure to SMILES.")
        return np.zeros((nbits,), dtype=np.float32)

    if isinstance(normalized, str):
        return derive_fingerprint_from_smiles(
            normalized,
            fingerprint_type=fingerprint_type,
            nbits=nbits,
            policy_invalid_smiles=policy_invalid,
        )

    valid_smiles = [x for x in normalized if x is not None]
    if len(valid_smiles) == 0:
        if policy_invalid == "raise":
            raise ValueError("No valid SMILES/InChI entries available for fingerprinting.")
        return np.zeros((0, nbits), dtype=np.float32)

    return derive_fingerprint_from_smiles(
        valid_smiles,
        fingerprint_type=fingerprint_type,
        nbits=nbits,
        policy_invalid_smiles=policy_invalid,
    )
