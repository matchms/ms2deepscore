from pathlib import Path
from typing import Tuple, Union
import h5py
import json
from tensorflow.python.keras.saving import hdf5_format

from ms2deepscore.SpectrumBinner import SpectrumBinner
from .SiameseModel import SiameseModel


def load_model(filename: Union[str, Path]) -> SiameseModel:
    """
    Load a MS2DeepScore model (SiameseModel) from file.
    
    Parameters
    ----------
    filename
        Filename. Expecting saved SiameseModel.
        
    """
    with h5py.File(filename, mode='r') as f:
        binner_json = f.attrs['spectrum_binner']
        keras_model = hdf5_format.load_model_from_hdf5(f)

    # Reconstitute spectrum_binner
    binner_dict = json.loads(binner_json)
    spectrum_binner = SpectrumBinner(binner_dict["number_of_bins"],
                                     binner_dict["mz_max"], binner_dict["mz_min"],
                                     binner_dict["peak_scaling"],
                                     binner_dict["allowed_missing_percentage"])
    spectrum_binner.peak_to_position = {int(key): value for key, value in binner_dict["peak_to_position"].items()}
    spectrum_binner.known_bins = binner_dict["known_bins"]

    # Extract parameters for SiameseModel
    embedding_dim = keras_model.layers[2].output_shape[1]
    base_dims = []
    for layer in keras_model.layers[2].layers:
        if "dense" in layer.name:
            base_dims.append(layer.output_shape[1])
        elif "dropout" in layer.name:
            dropout_rate = layer.rate

    model = SiameseModel(spectrum_binner, base_dims, embedding_dim, dropout_rate)
    # TODO: Now this creates a keras model in the SiameseModel.__init__() and then replaces this. Seems unefficient.
    model.base = keras_model.layers[2]
    model.model = keras_model
    return model