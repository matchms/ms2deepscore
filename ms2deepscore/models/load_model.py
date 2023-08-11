from pathlib import Path
from typing import Union
import h5py
from tensorflow import keras
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
    with h5py.File(filename, mode='r') as f:
        binner_json = f.attrs['spectrum_binner']
        keras_model = keras.models.load_model(f)

    spectrum_binner = SpectrumBinner.from_json(binner_json)
    return SiameseModel(spectrum_binner, keras_model=keras_model)
