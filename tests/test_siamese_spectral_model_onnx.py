import json
import numpy as np
import pytest
from matchms import Spectrum
from unittest.mock import MagicMock, patch

from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.models.SiameseSpectralModelONNX import (
    SiameseSpectralModelONNX,
    configure_onnx_providers,
    validate_onnx_session,
)
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_spectra():
    spectrum1 = Spectrum(
        mz=np.array([101, 202, 303.0]),
        intensities=np.array([0.1, 0.2, 1.0]),
        metadata={"precursor_mz": 222.2},
    )
    spectrum2 = Spectrum(
        mz=np.array([101.5, 202.5, 303.0]),
        intensities=np.array([0.1, 0.2, 1.0]),
        metadata={"precursor_mz": 333.3},
    )
    return [spectrum1, spectrum2]


@pytest.fixture
def model_settings():
    return SettingsMS2Deepscore(mz_bin_width=1.0)


@pytest.fixture
def model_settings_with_metadata():
    return SettingsMS2Deepscore(
        mz_bin_width=1.0,
        additional_metadata=[
            ("StandardScaler", {
                "metadata_field": "precursor_mz",
                "mean": 200.0,
                "standard_deviation": 250.0,
            })
        ],
        train_binning_layer=False,
    )


@pytest.fixture
def exported_onnx_model(tmp_path, model_settings):
    """Exports a SiameseSpectralModel to ONNX and returns the path."""
    pytorch_model = SiameseSpectralModel(model_settings)
    pytorch_model.export_to_onnx(tmp_path, model_name="test_model")
    return tmp_path / "test_model.onnx"


@pytest.fixture
def exported_onnx_model_with_metadata(tmp_path, model_settings_with_metadata):
    """Exports a SiameseSpectralModel with metadata to ONNX and returns the path."""
    pytorch_model = SiameseSpectralModel(model_settings_with_metadata)
    pytorch_model.export_to_onnx(tmp_path, model_name="test_model_meta")
    return tmp_path / "test_model_meta.onnx"


@pytest.fixture
def onnx_model(exported_onnx_model):
    """Returns a loaded SiameseSpectralModelONNX."""
    return SiameseSpectralModelONNX(exported_onnx_model)


# ---------------------------------------------------------------------------
# Loading & settings
# ---------------------------------------------------------------------------

def test_onnx_model_loads(exported_onnx_model):
    """Model loads without errors from a valid ONNX file."""
    model = SiameseSpectralModelONNX(exported_onnx_model)
    assert model.session is not None


def test_onnx_model_settings_loaded(onnx_model, model_settings):
    """Settings stored in the ONNX metadata are correctly deserialised."""
    assert isinstance(onnx_model.model_settings, SettingsMS2Deepscore)
    assert onnx_model.model_settings.mz_bin_width == model_settings.mz_bin_width
    assert onnx_model.model_settings.embedding_dim == model_settings.embedding_dim


# ---------------------------------------------------------------------------
# validate_onnx_session
# ---------------------------------------------------------------------------

def test_validate_onnx_session_valid(onnx_model):
    """validate_onnx_session does not raise for a correctly exported model."""
    validate_onnx_session(onnx_model.session)  # should not raise


def test_validate_onnx_session_missing_input():
    """validate_onnx_session raises when required input is missing."""
    session = MagicMock()
    session.get_inputs.return_value = []  # no inputs
    out = MagicMock()
    out.name = "embedding"
    session.get_outputs.return_value = [out]

    with pytest.raises(ValueError, match="spectra_tensors"):
        validate_onnx_session(session)


def test_validate_onnx_session_missing_output():
    """validate_onnx_session raises when required output is missing."""
    session = MagicMock()
    inp = MagicMock()
    inp.name = "spectra_tensors"
    session.get_inputs.return_value = [inp]
    session.get_outputs.return_value = []  # no outputs

    with pytest.raises(ValueError, match="embedding"):
        validate_onnx_session(session)


# ---------------------------------------------------------------------------
# compute_embedding_array
# ---------------------------------------------------------------------------

def test_compute_embedding_array_shape(onnx_model, dummy_spectra):
    """Embedding array has the correct shape."""
    embeddings = onnx_model.compute_embedding_array(dummy_spectra, progress_bar=False)
    assert embeddings.shape == (len(dummy_spectra), onnx_model.model_settings.embedding_dim)


def test_compute_embedding_array_dtype(onnx_model, dummy_spectra):
    """Embedding array is float32."""
    embeddings = onnx_model.compute_embedding_array(dummy_spectra, progress_bar=False)
    assert embeddings.dtype == np.float32


def test_compute_embedding_array_with_metadata(exported_onnx_model_with_metadata, dummy_spectra):
    """Embedding computation works with additional metadata inputs."""
    model = SiameseSpectralModelONNX(exported_onnx_model_with_metadata)
    embeddings = model.compute_embedding_array(dummy_spectra, progress_bar=False)
    assert embeddings.shape == (len(dummy_spectra), model.model_settings.embedding_dim)


def test_compute_embedding_array_batching(onnx_model, dummy_spectra):
    """Batch size 1 and batch size equal to n_spectra produce identical results."""
    emb_batch_1 = onnx_model.compute_embedding_array(dummy_spectra, batch_size=1, progress_bar=False)
    emb_full = onnx_model.compute_embedding_array(dummy_spectra, batch_size=len(dummy_spectra), progress_bar=False)
    np.testing.assert_allclose(emb_batch_1, emb_full, rtol=1e-5)


def test_compute_embedding_array_invalid_batch_size(onnx_model, dummy_spectra):
    """batch_size <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="batch_size"):
        onnx_model.compute_embedding_array(dummy_spectra, batch_size=0, progress_bar=False)


def test_compute_embedding_array_no_binning_layer(exported_onnx_model, dummy_spectra):
    """Forward pass works for models without a binning layer (default)."""
    model = SiameseSpectralModelONNX(exported_onnx_model)
    assert not model.model_settings.train_binning_layer
    embeddings = model.compute_embedding_array(dummy_spectra, progress_bar=False)
    assert embeddings.shape[0] == len(dummy_spectra)


# ---------------------------------------------------------------------------
# configure_onnx_providers
# ---------------------------------------------------------------------------

def test_configure_onnx_providers_default_includes_cpu():
    """CPU provider is always present as fallback."""
    providers = configure_onnx_providers()
    provider_names = [p if isinstance(p, str) else p[0] for p in providers]
    assert "CPUExecutionProvider" in provider_names


def test_configure_onnx_providers_cpu_is_last():
    """CPU provider is always the last fallback."""
    providers = configure_onnx_providers()
    last = providers[-1]
    name = last if isinstance(last, str) else last[0]
    assert name == "CPUExecutionProvider"


def test_configure_onnx_providers_openvino_fp16():
    """OpenVINO provider uses FP16 hint when precision=16."""
    with patch("ms2deepscore.models.SiameseSpectralModelONNX.ort") as mock_ort:
        mock_ort.get_available_providers.return_value = [
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = configure_onnx_providers(precision=16)

    ov_provider = next(p for p in providers if isinstance(p, tuple) and p[0] == "OpenVINOExecutionProvider")
    config = json.loads(ov_provider[1]["load_config"])
    assert config["GPU"]["INFERENCE_PRECISION_HINT"] == "f16"


def test_configure_onnx_providers_openvino_fp32():
    """OpenVINO provider uses FP32 hint when precision=32 (default)."""
    with patch("ms2deepscore.models.SiameseSpectralModelONNX.ort") as mock_ort:
        mock_ort.get_available_providers.return_value = [
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = configure_onnx_providers(precision=32)

    ov_provider = next(p for p in providers if isinstance(p, tuple) and p[0] == "OpenVINOExecutionProvider")
    config = json.loads(ov_provider[1]["load_config"])
    assert config["GPU"]["INFERENCE_PRECISION_HINT"] == "f32"


def test_configure_onnx_providers_no_openvino_without_hardware():
    """OpenVINO provider is absent when not in available providers."""
    with patch("ms2deepscore.models.SiameseSpectralModelONNX.ort") as mock_ort:
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        providers = configure_onnx_providers()

    provider_names = [p if isinstance(p, str) else p[0] for p in providers]
    assert "OpenVINOExecutionProvider" not in provider_names