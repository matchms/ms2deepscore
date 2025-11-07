from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json
import warnings
import numpy as np
import torch
from ms2deepscore.__version__ import __version__ as _MS2DS_VERSION


# -------------------------
# JSON helpers (no pickling)
# -------------------------

def _to_jsonable(obj: Any) -> Any:
    """Recursively convert settings/config to JSON-serializable primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        from enum import Enum
        if isinstance(obj, Enum):
            return obj.value
    except Exception:
        pass
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if hasattr(obj, "dict"):
        return _to_jsonable(obj.dict())
    if hasattr(obj, "to_dict"):
        return _to_jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return str(obj)

def _to_settings_json(settings_obj: Any) -> str:
    """Strictly JSON-encode settings to ensure safe loading later."""
    return json.dumps(_to_jsonable(settings_obj), ensure_ascii=False, sort_keys=True)

# -------------------------
# Legacy extraction
# -------------------------

def _extract_from_legacy_object(obj: Any) -> Tuple[Dict[str, torch.Tensor], str, str]:
    """
    Return (state_dict, settings_json, model_class) from a legacy artifact.
    Raises TypeError for unsupported shapes.
    """
    # Case 1: pickled model object (torch.save(model))
    if isinstance(obj, torch.nn.Module):
        if not hasattr(obj, "state_dict"):
            raise TypeError("Legacy model object has no state_dict().")
        state = obj.state_dict()
        # Commonly lives at obj.model_settings; fall back to __dict__ if needed.
        settings_obj = getattr(obj, "model_settings", getattr(obj, "settings", None))
        if settings_obj is None:
            # Last resort: empty settings (not ideal, but avoids blocking conversion)
            settings_json = json.dumps({}, sort_keys=True)
        else:
            settings_json = _to_settings_json(settings_obj)
        model_class = obj.__class__.__name__
        return state, settings_json, model_class

    # Case 2: dict-style legacy checkpoint
    if isinstance(obj, dict):
        # weights
        if "state_dict" in obj:
            state = obj["state_dict"]
        elif "model_state_dict" in obj:
            state = obj["model_state_dict"]
        else:
            raise TypeError("Legacy dict has no 'state_dict' or 'model_state_dict'.")

        # settings
        if "settings_json" in obj:
            settings_json = obj["settings_json"]
            if not isinstance(settings_json, str):
                settings_json = json.dumps(settings_json, ensure_ascii=False, sort_keys=True)
        elif "model_params_json" in obj:
            settings_json = obj["model_params_json"]
            if not isinstance(settings_json, str):
                settings_json = json.dumps(settings_json, ensure_ascii=False, sort_keys=True)
        elif "model_params" in obj:
            settings_json = json.dumps(_to_jsonable(obj["model_params"]), ensure_ascii=False, sort_keys=True)
        else:
            # best effort: no settings available
            settings_json = json.dumps({}, sort_keys=True)

        model_class = obj.get("model_class", "Unknown")
        return state, settings_json, model_class

    raise TypeError(
        "Unsupported legacy artifact. Expected pickled torch.nn.Module or dict-style checkpoint."
    )

# -------------------------
# Public converter
# -------------------------

def convert_legacy_checkpoint(
    legacy_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Convert a TRUSTED legacy ms2deepscore model file into a safe single-file checkpoint
    consumable with torch.load(..., weights_only=True).

    Parameters
    ----------
    legacy_path : str | Path
        Path to the legacy file (pickled model object or dict checkpoint).
    output_path : str | Path | None
        Path to write the converted file. Defaults to `<legacy_path>.converted.pt`.
    overwrite : bool
        Overwrite output_path if it exists.

    Returns
    -------
    Path
        The path of the converted safe checkpoint.

    Notes
    -----
    This function performs an UNSAFE load (weights_only=False) to read the legacy
    file. Only use on files from trusted sources.
    """
    legacy_path = Path(legacy_path)
    if output_path is None:
        output_path = legacy_path.with_suffix(legacy_path.suffix + ".converted.pt")
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use overwrite=True)")

    warnings.warn(
        "Loading legacy checkpoint with weights_only=False. "
        "Only do this for trusted files.",
        RuntimeWarning,
    )
    legacy_obj = torch.load(str(legacy_path), map_location="cpu", weights_only=False)

    state_dict, settings_json, model_class = _extract_from_legacy_object(legacy_obj)

    safe_ckpt = {
        "format": "ms2deepscore.safe.v1",
        "ms2deepscore_version": getattr(legacy_obj, "version", _MS2DS_VERSION),
        "model_class": model_class,
        "settings_json": settings_json,   # pure JSON string
        "state_dict": state_dict,         # tensors only
    }

    torch.save(safe_ckpt, str(output_path))
    return output_path
