import json
import platform
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from matchms import Spectrum
from tqdm.auto import tqdm
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.tensorize_spectra import tensorize_spectra_onnx


try:
    import openvino  # noqa: F401
except ImportError:
    openvino = None


class SiameseSpectralModelONNX:
    """
    SiameseSpectralModelONNX for inference with onnx runtime.
    Training is done via the SiameseSpectralModel.
    """

    def __init__(self, model_path: str | Path, providers: list | None = None, **kwargs):
        if providers is None:
            providers = configure_onnx_providers()

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        validate_onnx_session(self.session)

        validate_settings = kwargs.get("validate_settings", True)
        self.model_settings = self._load_settings(validate_settings)

    def _load_settings(self, validate_settings: bool = True) -> SettingsMS2Deepscore:
        """Extract and deserialize model settings from ONNX model metadata."""
        model_metadata = self.session.get_modelmeta().custom_metadata_map

        if "settings" not in model_metadata:
            raise ValueError("Model does not contain settings. These are required for inference.")

        settings_dict = json.loads(model_metadata["settings"])
        settings_dict["spectrum_file_path"] = None

        return SettingsMS2Deepscore(**settings_dict, validate_settings=validate_settings)

    def compute_embedding_array(
        self,
        spectra: list[Spectrum],
        batch_size: int = 1024,
        progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Compute the embeddings of all given spectra (list of matchms Spectrum objects) using onnxruntime.

        Parameters
        ----------
        spectra:
            A list of matchms spectra.
        batch_size:
            The batch size for inference, defaults to 1024.
        progress_bar:
            Whether to display a progress bar during embedding computation.
        """
        settings = self.model_settings

        x_binned, x_metadata = tensorize_spectra_onnx(spectra, settings)

        onnx_inputs = {inp.name for inp in self.session.get_inputs()}
        has_metadata = "metadata_tensors" in onnx_inputs
        num_samples = x_binned.shape[0]
        embedding_dim = settings.embedding_dim
        embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)

        with tqdm(total=num_samples, desc="Computing embeddings", unit="spectrum", disable=not progress_bar) as pbar:
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)

                input_feed = {"spectra_tensors": x_binned[start:end]}
                if has_metadata:
                    input_feed["metadata_tensors"] = x_metadata[start:end]

                embeddings[start:end] = self.session.run(["embedding"], input_feed)[0]

                pbar.update(end - start)

        return embeddings


def configure_onnx_providers(precision: Literal[16, 32] = 32) -> list:
    """
    Reads and configures all onnxruntime backend providers available to the system.

    Parameters
    ----------
    precision:
        Inference precision (16 or 32). Used for OpenVINO if no openvino_config is provided.

    Returns
    -------
    List of configured providers to be used for onnxruntime inference.
    """
    available = ort.get_available_providers()
    providers = []

    # CUDA
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    # macOS -> use CoreML
    if platform.system() == "Darwin":
        major, minor, *_ = map(int, platform.mac_ver()[0].split("."))
        if (major, minor) >= (12, 0):
            providers.append(
                (
                    "CoreMLExecutionProvider",
                    {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
                )
            )

    # Intel -> OpenVino
    if "OpenVINOExecutionProvider" in available:
        prec = "f16" if precision == 16 else "f32"
        options = {
            "device_type": "GPU",
            "load_config": json.dumps({"GPU": {"INFERENCE_PRECISION_HINT": prec}}),
        }

        providers.append(("OpenVINOExecutionProvider", options))

    # Fallback
    providers.append("CPUExecutionProvider")

    return providers


def validate_onnx_session(session: ort.InferenceSession) -> None:
    """Raise early if the ONNX model does not match the expected interface."""
    input_names = {inp.name for inp in session.get_inputs()}
    output_names = {out.name for out in session.get_outputs()}

    missing_inputs = {"spectra_tensors"} - input_names
    missing_outputs = {"embedding"} - output_names

    if missing_inputs or missing_outputs:
        raise ValueError(
            f"model does not match expected interface. Missing inputs: {missing_inputs}. Missing outputs: {missing_outputs}."
        )
