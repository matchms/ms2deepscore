from typing import Any
from pathlib import Path
import json
import numpy as np


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert settings to JSON-safe primitives."""
    # Basic primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # numpy scalars
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    # sequences
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # mappings
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pathlib Paths, enums, etc.
    try:
        from enum import Enum
        if isinstance(obj, Enum):
            return obj.value
    except Exception:
        pass
    if isinstance(obj, (Path,)):
        return str(obj)
    # pydantic v2
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    # pydantic v1 / dataclass-like
    if hasattr(obj, "dict"):
        return _to_jsonable(obj.dict())
    if hasattr(obj, "to_dict"):
        return _to_jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    # last resort: stringify
    return str(obj)


def _settings_to_json(settings: Any) -> str:
    """Strictly JSON-encode settings to guarantee safe loading."""
    plain = _to_jsonable(settings)
    return json.dumps(plain, ensure_ascii=False, sort_keys=True)
