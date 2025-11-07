from typing import Union, Dict, Optional, Any
import json
from pathlib import Path
import warnings
import numpy as np
import torch
from ms2deepscore.__version__ import __version__
from ms2deepscore.models.EmbeddingEvaluatorModel import EmbeddingEvaluationModel
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.models.LinearEmbeddingEvaluation import LinearModel
from ms2deepscore.SettingsMS2Deepscore import (
    SettingsEmbeddingEvaluator,
    SettingsMS2Deepscore,
)


# ---------- internal helpers ----------

def _torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_ckpt_safe(filename: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a checkpoint using PyTorch's restricted unpickler (weights_only=True).
    The file must contain only tensors and simple Python primitives.
    """
    try:
        return torch.load(str(filename), map_location=_torch_device(), weights_only=True)
    except TypeError:
        # Older torch may not support weights_only -> fall back but still expect only primitives/tensors.
        ckpt = torch.load(str(filename), map_location=_torch_device())
        if not isinstance(ckpt, dict):
            raise
        return ckpt

def _extract_settings_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support current format and a couple of earlier safe variants.
    Priority:
      1) settings_json (preferred, current format)
      2) model_params_json (earlier safe format)
      3) model_params (earlier safe format with a plain dict)
    """
    if "settings_json" in ckpt:
        return json.loads(ckpt["settings_json"])
    if "model_params_json" in ckpt:
        return json.loads(ckpt["model_params_json"])
    if "model_params" in ckpt:
        params = ckpt["model_params"]
        if not isinstance(params, dict):
            raise TypeError("Expected 'model_params' to be a dict.")
        return params
    raise KeyError(
        "No settings found. Expected 'settings_json' (preferred) or 'model_params_json' / 'model_params'."
    )

def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Support current key 'state_dict' and older 'model_state_dict'.
    """
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    raise KeyError("No weights found. Expected 'state_dict' or 'model_state_dict'.")

def _maybe_warn_version(ckpt: Dict[str, Any]) -> None:
    v = ckpt.get("ms2deepscore_version") or ckpt.get("version")
    if v and v != __version__:
        warnings.warn(
            f"Model was saved with ms2deepscore {v}, but you're running {__version__}. "
            "Consider updating either the model or the library if you hit incompatibilities."
        )

def _convert_legacy_if_requested(
    obj: Any,
    *,
    convert_path: Optional[Union[str, Path]],
) -> Optional[Path]:
    """
    If a legacy artifact was loaded unsafely (pickled object or dict), convert it
    to the new safe format and write it to 'convert_path' if provided.
    Returns the path written, or None if not written.
    """
    if convert_path is None:
        return None

    out_path = Path(convert_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Two conversion paths:
    # 1) If it's an nn.Module with .state_dict and .model_settings
    if isinstance(obj, torch.nn.Module) and hasattr(obj, "state_dict") and hasattr(obj, "model_settings"):
        # Build a minimal safe checkpoint (match the new save())
        from . import _settings_to_json  # re-use from your module where save() lives
        safe_ckpt = {
            "format": "ms2deepscore.safe.v1",
            "ms2deepscore_version": getattr(obj, "version", __version__),
            "model_class": obj.__class__.__name__,
            "settings_json": _settings_to_json(obj.model_settings),
            "state_dict": obj.state_dict(),
        }
        torch.save(safe_ckpt, str(out_path))
        return out_path

    # 2) If it's a dict-like legacy checkpoint with params + state_dict
    if isinstance(obj, dict) and ("model_state_dict" in obj or "state_dict" in obj):
        # Try to normalize to the new shape
        try:
            params = _extract_settings_dict(obj)
        except Exception:
            params = obj.get("model_params", {})
        # JSON encode params to ensure safety
        settings_json = json.dumps(params, ensure_ascii=False, sort_keys=True)

        state_dict = _extract_state_dict(obj)
        safe_ckpt = {
            "format": "ms2deepscore.safe.v1",
            "ms2deepscore_version": obj.get("ms2deepscore_version") or obj.get("version") or __version__,
            "model_class": obj.get("model_class", "Unknown"),
            "settings_json": settings_json,
            "state_dict": state_dict,
        }
        torch.save(safe_ckpt, str(out_path))
        return out_path

    # Otherwise we don't know how to convert
    return None


# ---------- public API ----------

def load_model(
    filename: Union[str, Path],
    *,
    allow_legacy: bool = False,
    convert_legacy_to: Optional[Union[str, Path]] = None,
) -> SiameseSpectralModel:
    """
    Load a SiameseSpectralModel.

    Normal path:
      1) Safe-load checkpoint (weights_only=True).
      2) Parse settings_json -> SettingsMS2Deepscore.
      3) Instantiate SiameseSpectralModel(settings=...) then load state_dict.

    Legacy path (only if allow_legacy=True):
      - Attempt torch.load(weights_only=False) and either:
         * return the nn.Module directly (if saved whole), or
         * adapt a dict-like legacy checkpoint to the new format.
      - If convert_legacy_to is given, also write a converted, safe checkpoint.

    Parameters
    ----------
    filename : str | Path
    allow_legacy : bool, default False
        Permit unsafe pickle loading for truly old files. Only use for trusted sources.
    convert_legacy_to : Optional[str | Path]
        If given and a legacy artifact is loaded, write an equivalent safe checkpoint here.
    """
    device = _torch_device()

    # --- preferred safe path
    try:
        ckpt = _load_ckpt_safe(filename)
        _maybe_warn_version(ckpt)
        params = _extract_settings_dict(ckpt)
        state_dict = _extract_state_dict(ckpt)

        settings = SettingsMS2Deepscore(**params, validate_settings=False)
        model = SiameseSpectralModel(settings=settings)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as safe_err:
        if not allow_legacy:
            raise RuntimeError(
                "Failed to load safely. If this is a trusted legacy file, call with allow_legacy=True."
            ) from safe_err

    # --- legacy fallback (unsafe; only for trusted files)
    warnings.warn(
        "Using UNSAFE legacy loading (weights_only=False). Only do this for trusted files.",
        RuntimeWarning,
    )
    legacy_obj = torch.load(str(filename), map_location=device, weights_only=False)

    # If the whole module was saved, just use it.
    if isinstance(legacy_obj, torch.nn.Module):
        legacy_obj.eval()
        _convert_legacy_if_requested(legacy_obj, convert_path=convert_legacy_to)
        return legacy_obj

    # If it looks like a legacy dict checkpoint, normalize and build the model
    if isinstance(legacy_obj, dict):
        try:
            params = _extract_settings_dict(legacy_obj)
            state_dict = _extract_state_dict(legacy_obj)
        except Exception as err:
            raise TypeError("Unrecognized legacy checkpoint structure.") from err

        settings = SettingsMS2Deepscore(**params, validate_settings=False)
        model = SiameseSpectralModel(settings=settings)
        model.load_state_dict(state_dict)
        model.eval()

        _convert_legacy_if_requested(legacy_obj, convert_path=convert_legacy_to)
        return model

    raise TypeError("Legacy artifact is neither a torch.nn.Module nor a compatible dict checkpoint.")


def load_embedding_evaluator(
    filename: Union[str, Path],
    *,
    allow_legacy: bool = False,
    convert_legacy_to: Optional[Union[str, Path]] = None,
) -> EmbeddingEvaluationModel:
    """
    Load an EmbeddingEvaluationModel with the same safe-first, legacy-optional policy.
    """
    device = _torch_device()

    # --- preferred safe path
    try:
        ckpt = _load_ckpt_safe(filename)
        _maybe_warn_version(ckpt)
        params = _extract_settings_dict(ckpt)
        state_dict = _extract_state_dict(ckpt)

        settings = SettingsEmbeddingEvaluator(**params)
        model = EmbeddingEvaluationModel(settings=settings)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as safe_err:
        if not allow_legacy:
            raise RuntimeError(
                "Failed to load safely. If this is a trusted legacy file, call with allow_legacy=True."
            ) from safe_err

    # --- legacy fallback (unsafe; only for trusted files)
    warnings.warn(
        "Using UNSAFE legacy loading (weights_only=False). Only do this for trusted files.",
        RuntimeWarning,
    )
    legacy_obj = torch.load(str(filename), map_location=device, weights_only=False)

    if isinstance(legacy_obj, torch.nn.Module):
        legacy_obj.eval()
        _convert_legacy_if_requested(legacy_obj, convert_path=convert_legacy_to)
        return legacy_obj

    if isinstance(legacy_obj, dict):
        try:
            params = _extract_settings_dict(legacy_obj)
            state_dict = _extract_state_dict(legacy_obj)
        except Exception as err:
            raise TypeError("Unrecognized legacy checkpoint structure.") from err

        settings = SettingsEmbeddingEvaluator(**params)
        model = EmbeddingEvaluationModel(settings=settings)
        model.load_state_dict(state_dict)
        model.eval()

        _convert_legacy_if_requested(legacy_obj, convert_path=convert_legacy_to)
        return model

    raise TypeError("Legacy artifact is neither a torch.nn.Module nor a compatible dict checkpoint.")


def load_linear_model(filepath):
    """Load a LinearModel from json.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        model_params = json.load(f)

    loaded_model = LinearModel(model_params["degree"])
    loaded_model.model.coef_ = np.array(model_params['coef'])
    loaded_model.model.intercept_ = np.array(model_params['intercept'])
    loaded_model.poly._min_degree = model_params["min_degree"]
    loaded_model.poly._max_degree = model_params["max_degree"]
    loaded_model.poly._n_out_full = model_params["_n_out_full"]
    loaded_model.poly.n_output_features_ = model_params["n_output_features_"]
    return loaded_model
