import json
from pathlib import Path
import pytest
import torch
from ms2deepscore.models import EmbeddingEvaluationModel, SiameseSpectralModel
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore, SettingsEmbeddingEvaluator
from ms2deepscore.models import load_model, load_embedding_evaluator
from ms2deepscore.models.io_utils import _to_jsonable


def _dummy_siamese_model():
    """Helper function to create a dummy SiameseSpectralModel."""
    settings = SettingsMS2Deepscore(base_dims=(100, 100), embedding_dim=10)
    model = SiameseSpectralModel(settings=settings)
    return model


def _create_safe_checkpoint(model):
    """Helper function to save a model as a safe checkpoint."""
    model.eval()  # Ensure model is in eval mode
    checkpoint = {
        "format": "ms2deepscore.safe.v1",
        "ms2deepscore_version": "2.5.5",
        "model_class": "SiameseSpectralModel",
        "settings_json": json.dumps(model.model_settings.__dict__, default=_to_jsonable),
        "state_dict": model.state_dict(),
    }
    return checkpoint


@pytest.fixture
def safe_checkpoint_siamese(tmp_path):
    """Fixture to generate a safe checkpoint for SiameseSpectralModel."""
    model = _dummy_siamese_model()
    checkpoint = _create_safe_checkpoint(model)
    checkpoint_path = tmp_path / "safe_model.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def legacy_checkpoint_siamese(tmp_path):
    """Fixture to generate a legacy checkpoint (pickled model)."""
    model = _dummy_siamese_model()
    legacy_path = tmp_path / "legacy_model.pt"
    torch.save(model, legacy_path)
    return legacy_path


@pytest.fixture
def legacy_dict_checkpoint(tmp_path):
    """Fixture to generate a legacy dict checkpoint."""
    model = _dummy_siamese_model()
    legacy_dict = {
        "model_state_dict": model.state_dict(),
        "model_params": model.model_settings.__dict__,
        "version": "9.9.9",
    }
    legacy_dict_path = tmp_path / "legacy_dict_model.pt"
    torch.save(legacy_dict, legacy_dict_path)
    return legacy_dict_path


# ----------------------
# Helpers (EmbeddingEvaluator)
# ----------------------
def _dummy_evaluator_model():
    # Stick to small, valid defaults
    eval_settings = SettingsEmbeddingEvaluator(
        evaluator_distribution_size=64,
        evaluator_num_filters=8,
        evaluator_depth=1,
        evaluator_kernel_size=5,
        mini_batch_size=2,
        batches_per_iteration=3,
        learning_rate=0.001,
        num_epochs=1,
    )
    return EmbeddingEvaluationModel(settings=eval_settings)

def _create_safe_ckpt_evaluator(model):
    model.eval()
    return {
        "format": "ms2deepscore.safe.v1",
        "ms2deepscore_version": "2.5.5",
        "model_class": "EmbeddingEvaluationModel",
        "settings_json": json.dumps(model.settings.__dict__, default=_to_jsonable),
        "state_dict": model.state_dict(),
    }

@pytest.fixture
def safe_checkpoint_evaluator(tmp_path: Path):
    model = _dummy_evaluator_model()
    ckpt = _create_safe_ckpt_evaluator(model)
    p = tmp_path / "safe_evaluator.pt"
    torch.save(ckpt, p)
    return p

@pytest.fixture
def legacy_checkpoint_evaluator(tmp_path: Path):
    model = _dummy_evaluator_model()
    p = tmp_path / "legacy_evaluator.pt"
    torch.save(model, p)  # pickled module
    return p

@pytest.fixture
def legacy_dict_checkpoint_evaluator(tmp_path: Path):
    model = _dummy_evaluator_model()
    p = tmp_path / "legacy_evaluator_dict.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": model.settings.__dict__,
            "version": "9.9.9",
        },
        p,
    )
    return p


def test_load_safe_checkpoint(safe_checkpoint_siamese):
    """Test loading a safe checkpoint."""
    model = load_model(safe_checkpoint_siamese)
    assert isinstance(model, SiameseSpectralModel)
    assert model.model_settings.base_dims == (100, 100)
    assert model.model_settings.embedding_dim == 10

    # Ensure that state_dict() is available and has content
    state_dict = model.state_dict()
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0


def test_load_legacy_model(legacy_checkpoint_siamese):
    """Test loading a legacy pickled model with allow_legacy=True."""
    model = load_model(legacy_checkpoint_siamese, allow_legacy=True)
    assert isinstance(model, SiameseSpectralModel)
    assert model.model_settings.base_dims == (100, 100)
    assert model.model_settings.embedding_dim == 10


def test_load_legacy_dict_model(legacy_dict_checkpoint):
    """Test loading a legacy dict checkpoint with allow_legacy=True."""
    model = load_model(legacy_dict_checkpoint, allow_legacy=True)
    assert isinstance(model, SiameseSpectralModel)
    assert model.model_settings.base_dims == (100, 100)
    assert model.model_settings.embedding_dim == 10


def test_load_legacy_with_version_warning(legacy_dict_checkpoint):
    """Test that a version mismatch triggers a warning."""
    with pytest.warns(Warning) as record:
        load_model(legacy_dict_checkpoint, allow_legacy=True)
    assert len(record) > 0
    assert "Using UNSAFE legacy loading (weights_only=False)" in str(record[0].message)


def test_load_legacy_and_convert(legacy_checkpoint_siamese, tmp_path):
    """Test legacy model conversion into a safe format."""
    convert_path = tmp_path / "converted_model.pt"
    _ = load_model(legacy_checkpoint_siamese, allow_legacy=True, convert_legacy_to=convert_path)
    
    # Check that the new file was created
    assert convert_path.exists()
    
    # Load the converted model to verify it's safe
    converted_ckpt = torch.load(str(convert_path), map_location="cpu", weights_only=True)
    assert "settings_json" in converted_ckpt
    assert "state_dict" in converted_ckpt


def test_load_model_with_invalid_checkpoint(tmp_path):
    """Test loading from an invalid checkpoint (neither safe nor legacy)."""
    invalid_checkpoint = tmp_path / "invalid_model.pt"
    with open(invalid_checkpoint, "w") as f:
        f.write("Not a valid checkpoint")

    with pytest.raises(RuntimeError):
        load_model(invalid_checkpoint)


@pytest.mark.parametrize("allow_legacy", [True, False])
def test_load_model_with_allow_legacy_param(legacy_dict_checkpoint, allow_legacy):
    """Test that allow_legacy param works with the model loader."""
    if allow_legacy:
        model = load_model(legacy_dict_checkpoint, allow_legacy=True)
        assert isinstance(model, SiameseSpectralModel)
    else:
        with pytest.raises(RuntimeError):
            load_model(legacy_dict_checkpoint, allow_legacy=False)


def test_siamese_legacy_pickled_load(legacy_checkpoint_siamese):
    model = load_model(legacy_checkpoint_siamese, allow_legacy=True)
    assert isinstance(model, SiameseSpectralModel)
    sd = model.state_dict()
    assert isinstance(sd, dict) and len(sd) > 0


def test_siamese_legacy_dict_load(legacy_dict_checkpoint):
    model = load_model(legacy_dict_checkpoint, allow_legacy=True)
    assert isinstance(model, SiameseSpectralModel)
    assert model.model_settings.base_dims == (100, 100)
    assert model.model_settings.embedding_dim == 10


def test_siamese_legacy_warns_and_converts(legacy_checkpoint_siamese, tmp_path: Path):
    convert_path = tmp_path / "converted_siamese.pt"
    with pytest.warns(RuntimeWarning) as rec:
        _ = load_model(legacy_checkpoint_siamese, allow_legacy=True, convert_legacy_to=convert_path)
    assert any("UNSAFE legacy loading" in str(w.message) for w in rec)
    assert convert_path.exists()
    # Converted file is safe-loadable
    ckpt = torch.load(str(convert_path), map_location="cpu", weights_only=True)
    assert "settings_json" in ckpt and "state_dict" in ckpt


@pytest.mark.parametrize("allow_legacy", [True, False])
def test_siamese_legacy_gate(legacy_dict_checkpoint, allow_legacy):
    if allow_legacy:
        model = load_model(legacy_dict_checkpoint, allow_legacy=True)
        assert isinstance(model, SiameseSpectralModel)
    else:
        with pytest.raises(RuntimeError):
            load_model(legacy_dict_checkpoint, allow_legacy=False)


# ----------------------
# EmbeddingEvaluator legacy route tests
# ----------------------

def test_evaluator_safe_load(safe_checkpoint_evaluator):
    model = load_embedding_evaluator(safe_checkpoint_evaluator)
    assert isinstance(model, EmbeddingEvaluationModel)
    sd = model.state_dict()
    assert isinstance(sd, dict) and len(sd) > 0


def test_evaluator_legacy_pickled_load(legacy_checkpoint_evaluator):
    model = load_embedding_evaluator(legacy_checkpoint_evaluator, allow_legacy=True)
    assert isinstance(model, EmbeddingEvaluationModel)
    sd = model.state_dict()
    assert isinstance(sd, dict) and len(sd) > 0


def test_evaluator_legacy_dict_load(legacy_dict_checkpoint_evaluator):
    model = load_embedding_evaluator(legacy_dict_checkpoint_evaluator, allow_legacy=True)
    assert isinstance(model, EmbeddingEvaluationModel)
    # spot-check a couple of default/known fields exist
    assert hasattr(model.settings, "evaluator_distribution_size")
    assert hasattr(model.settings, "evaluator_num_filters")


def test_evaluator_legacy_warns_and_converts(legacy_checkpoint_evaluator, tmp_path: Path):
    convert_path = tmp_path / "converted_evaluator.pt"
    with pytest.warns(RuntimeWarning) as rec:
        _ = load_embedding_evaluator(legacy_checkpoint_evaluator, allow_legacy=True, convert_legacy_to=convert_path)
    assert any("UNSAFE legacy loading" in str(w.message) for w in rec)


def test_evaluator_legacy_gate_allows_when_true(legacy_dict_checkpoint_evaluator):
    # dict-style legacy is fine when allow_legacy=True
    model = load_embedding_evaluator(legacy_dict_checkpoint_evaluator, allow_legacy=True)
    assert isinstance(model, EmbeddingEvaluationModel)


def test_evaluator_legacy_gate_blocks_when_false(legacy_checkpoint_evaluator):
    # pickled module triggers safe-load failure; with allow_legacy=False we expect RuntimeError
    with pytest.raises(RuntimeError):
        load_embedding_evaluator(legacy_checkpoint_evaluator, allow_legacy=False)
