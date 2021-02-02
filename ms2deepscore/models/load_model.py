from pathlib import Path
from typing import Union
import h5py
import json
from tensorflow.python.keras.saving import hdf5_format

from ms2deepscore.SpectrumBinner import SpectrumBinner
from .SiameseModel import SiameseModel


def load_model(filename: Union[str, Path]) -> SiameseModel:
    """
    Load a MS2DeepScore model (SiameseModel) from file.

    For example:

    .. code-block:: python

        from ms2deepscore.models import load_model
        model = load_model("model_file_xyz.hdf5")

    Parameters
    ----------
    filename
        Filename. Expecting saved SiameseModel.

    """
    def binner_from_json(binner_json):
        # Reconstitute spectrum_binner from json
        binner_dict = json.loads(binner_json)
        spectrum_binner = SpectrumBinner(binner_dict["number_of_bins"],
                                         binner_dict["mz_max"], binner_dict["mz_min"],
                                         binner_dict["peak_scaling"],
                                         binner_dict["allowed_missing_percentage"])
        spectrum_binner.peak_to_position = {int(key): value for key, value in binner_dict["peak_to_position"].items()}
        spectrum_binner.known_bins = binner_dict["known_bins"]
        return spectrum_binner

    with h5py.File(filename, mode='r') as f:
        binner_json = f.attrs['spectrum_binner']
        keras_model = hdf5_format.load_model_from_hdf5(f)

    spectrum_binner = binner_from_json(binner_json)
    return SiameseModel(spectrum_binner, keras_model=keras_model)
