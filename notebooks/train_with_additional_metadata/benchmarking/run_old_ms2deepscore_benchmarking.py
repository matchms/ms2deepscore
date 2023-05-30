import os
import pickle
# from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
from pathlib import Path
from typing import Union
import h5py
from tensorflow import keras

from ms2deepscore.SpectrumBinner import SpectrumBinner
from ms2deepscore.models.SiameseModel import SiameseModel


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


def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def create_all_plots(model_folder_name):
    data_dir = "../../../../data/"
    model_folder = f"../../../../data/trained_models/{model_folder_name}"

    positive_validation_spectra = load_pickled_file(os.path.join(
        data_dir, "training_and_validation_split", "positive_validation_spectra.pickle"))[:10]

    # Create benchmarking results folder
    benchmarking_results_folder = os.path.join(model_folder, "benchmarking_results")
    if not os.path.exists(benchmarking_results_folder):
        assert not os.path.isfile(benchmarking_results_folder), "The folder specified is a file"
        os.mkdir(benchmarking_results_folder)

    # Load in MS2Deepscore model
    ms2deepscore_model = load_model(os.path.join(model_folder, "ms2deepscore_model.hdf5"))
    similarity_score = MS2DeepScore(ms2deepscore_model)

    # Create predictions
    predictions_file_name = os.path.join(benchmarking_results_folder,
                                         f"postive_positive_a_few_predictions_ms2deepscore.pickle")
    predictions = similarity_score.matrix(positive_validation_spectra, positive_validation_spectra,
                                          is_symmetric=True)
    print(predictions)
    # save_pickled_file(predictions, predictions_file_name)


if __name__ == "__main__":
    create_all_plots("ms2ds_model_original_paper")
