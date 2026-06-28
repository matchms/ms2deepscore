import json
import platform
import numpy as np
from matchms import Spectrum
from tqdm.auto import tqdm


try:
    import openvino  # noqa: F401
except ImportError:
    openvino = None

import onnxruntime as ort
from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.tensorize_spectra import tensorize_spectra_onnx


def configure_onnx_providers() -> list:
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
        providers.append(("OpenVINOExecutionProvider", {"device_type": "GPU"}))

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
            f"model does not match expected interface. Missing inputs:  {missing_inputs}. Missing outputs: {missing_outputs}."
        )


def compute_embedding_array_onnx(
    onnx_session: ort.InferenceSession,
    spectra: list[Spectrum],
    settings: SettingsMS2Deepscore = None,
    batch_size: int = 1024,
    progress_bar: bool = True,
) -> np.ndarray:
    """
    Compute the embeddings of all given spectra (list of matchms Spectrum objects) using onnxruntime.

    Parameters
    ----------
    onnx_session:
        A onnx runtime session with loaded onnx model and attached providers.
    spectra:
        A list of matchms spectra.
    settings:
        A optional SettingsMS2Deepscore object. If not present is derived from the onnx model.
    batch_size:
        The batch size for inference, defaults to 1024.
    progress_bar:
        Whether to display a progress bar during embedding computation.
    """
    # validate_onnx_session(onnx_session)
    if not settings:
        model_metadata = onnx_session.get_modelmeta().custom_metadata_map
        settings_dict = json.loads(model_metadata["settings"])
        settings_dict["spectrum_file_path"] = None
        settings = SettingsMS2Deepscore(**settings_dict)

    x_binned, x_metadata = tensorize_spectra_onnx(spectra, settings)

    has_metadata = x_metadata.shape[1] > 0
    num_samples = x_binned.shape[0]
    embedding_dim = settings.embedding_dim
    embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)

    with tqdm(total=num_samples, desc="Computing embeddings", unit="spectrum", disable=not progress_bar) as pbar:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            input_feed = {"spectra_tensors": x_binned[start:end]}
            if has_metadata:
                input_feed["metadata_tensors"] = x_metadata[start:end]

            embeddings[start:end] = onnx_session.run(["embedding"], input_feed)[0]

            pbar.update(end - start)

    return embeddings
