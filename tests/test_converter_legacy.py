import json
from pathlib import Path
import pytest
import torch
from ms2deepscore.models import convert_legacy_checkpoint
from ms2deepscore.models.__model_format__ import __model_format__


def _dummy_module():
    """Create a small torch module and attach model_settings for legacy pickled-model tests."""
    m = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    # Attach something that looks like ms2deepscore settings
    m.model_settings = {"base_dims": (4, 3, 2), "embedding_dim": 2, "ionisation_mode": "positive"}
    return m


def _state_dicts_equal(a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a.keys():
        ta, tb = a[k], b[k]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            return False
        if ta.shape != tb.shape:
            return False
        if not torch.allclose(ta.cpu(), tb.cpu()):
            return False
    return True


def _assert_safe_checkpoint_structure(ckpt: dict):
    # top-level keys
    assert "format" in ckpt
    assert ckpt["format"].startswith("ms2deepscore.safe.")
    assert "ms2deepscore_version" in ckpt
    assert "model_class" in ckpt
    assert "settings_json" in ckpt
    assert "state_dict" in ckpt

    # types
    assert isinstance(ckpt["settings_json"], str)
    # must be valid JSON
    params = json.loads(ckpt["settings_json"])
    assert isinstance(params, dict)

    # state_dict must be tensor-only dict
    assert isinstance(ckpt["state_dict"], dict)
    assert all(isinstance(v, torch.Tensor) for v in ckpt["state_dict"].values())


def test_convert_pickled_module_to_safe(tmp_path: Path):
    legacy_path = tmp_path / "legacy_model_pickled.pt"

    # Create and save a legacy-style pickled module
    m = _dummy_module()
    torch.save(m, legacy_path)

    out = convert_legacy_checkpoint(legacy_path)
    assert out.exists()
    assert out.name.endswith(".converted.pt")

    # Safe-load with weights_only=True (PyTorch >= 2.6)
    ckpt = torch.load(str(out), map_location="cpu", weights_only=True)
    _assert_safe_checkpoint_structure(ckpt)

    # Weights match
    assert _state_dicts_equal(ckpt["state_dict"], m.state_dict())


def test_convert_legacy_dict_with_model_state_dict(tmp_path: Path):
    legacy_path = tmp_path / "legacy_dict.pt"

    m = _dummy_module()
    legacy = {
        "model_state_dict": m.state_dict(),
        "model_params": {"base_dims": (4, 3, 2), "embedding_dim": 2},
        "version": "9.9.9",
    }
    torch.save(legacy, legacy_path)

    out = convert_legacy_checkpoint(legacy_path)
    ckpt = torch.load(str(out), map_location="cpu", weights_only=True)

    _assert_safe_checkpoint_structure(ckpt)
    # Version should come from legacy dict's "version" if present
    assert ckpt["ms2deepscore_version"] == "9.9.9"
    assert ckpt["format"] == __model_format__
    assert _state_dicts_equal(ckpt["state_dict"], m.state_dict())


def test_convert_legacy_dict_with_state_dict_key(tmp_path: Path):
    legacy_path = tmp_path / "legacy_dict_state_dict.pt"

    m = _dummy_module()
    legacy = {
        "state_dict": m.state_dict(),  # alternate key
        "model_params": {"base_dims": (4, 3, 2), "embedding_dim": 2},
    }
    torch.save(legacy, legacy_path)

    out = convert_legacy_checkpoint(legacy_path, overwrite=True)  # explicit overwrite allowed
    ckpt = torch.load(str(out), map_location="cpu", weights_only=True)

    _assert_safe_checkpoint_structure(ckpt)
    assert _state_dicts_equal(ckpt["state_dict"], m.state_dict())


def test_default_output_path_and_overwrite_behavior(tmp_path: Path):
    legacy_path = tmp_path / "legacy_pickled.pt"
    m = _dummy_module()
    torch.save(m, legacy_path)

    # 1st conversion -> creates <legacy>.converted.pt
    out1 = convert_legacy_checkpoint(legacy_path)
    assert (out1 == tmp_path / "legacy_pickled.converted.pt")
    assert out1.exists()

    # 2nd conversion without overwrite -> should raise
    with pytest.raises(FileExistsError):
        convert_legacy_checkpoint(legacy_path)

    # 3rd conversion with overwrite=True -> should succeed
    out3 = convert_legacy_checkpoint(legacy_path, overwrite=True)
    assert out3 == out1
    assert out3.exists()


def test_explicit_output_path(tmp_path: Path):
    legacy_path = tmp_path / "legacy_any.pt"
    explicit_out = tmp_path / "converted_safe.pt"

    m = _dummy_module()
    torch.save(m, legacy_path)

    out = convert_legacy_checkpoint(legacy_path, output_path=explicit_out, overwrite=True)
    assert out == explicit_out
    ckpt = torch.load(str(out), map_location="cpu", weights_only=True)
    _assert_safe_checkpoint_structure(ckpt)


def test_settings_json_is_parsable_and_stable(tmp_path: Path):
    legacy_path = tmp_path / "legacy_model_pickled.pt"
    m = _dummy_module()
    torch.save(m, legacy_path)

    out = convert_legacy_checkpoint(legacy_path, overwrite=True)
    ckpt = torch.load(str(out), map_location="cpu", weights_only=True)

    # Ensure JSON parses and contains keys we put in model_settings
    settings = json.loads(ckpt["settings_json"])
    assert settings.get("base_dims") in [(4, 3, 2), [4, 3, 2]]  # JSON may store lists
    assert settings.get("embedding_dim") == 2
