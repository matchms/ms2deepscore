import os
from pathlib import Path
import pytest
import pickle
from ms2deepscore.models import SiameseModel
from ms2deepscore.models.load_model import load_model as load_ms2deepscore_model
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from matchms.importing import load_from_mgf
from ms2deepscore.train_new_model.calculate_tanimoto_matrix import calculate_tanimoto_scores_unique_inchikey

TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def test_train_ms2ds_model(tmp_path):
    spectra = list(load_from_mgf(os.path.join(TEST_RESOURCES_PATH, "pesticides_processed.mgf")))
    model_file_name = os.path.join(tmp_path, "ms2deepscore_model.hdf5")
    epochs = 2
    tanimoto_df = calculate_tanimoto_scores_unique_inchikey(spectra, spectra)
    history = train_ms2ds_model(spectra, spectra, tanimoto_df, model_file_name, epochs)
    assert os.path.isfile(model_file_name), "Expecte ms2ds model to be created and saved"
    ms2ds_model = load_ms2deepscore_model(model_file_name)
    assert isinstance(ms2ds_model, SiameseModel), "Expected a siamese model"
    assert isinstance(history, dict), "expected history to be a dictionary"
    assert list(history.keys()) == ['loss', 'mae', 'root_mean_squared_error', 'val_loss', 'val_mae', 'val_root_mean_squared_error']
    for scores in history.values():
        assert len(scores) == epochs, "expected the number of losses in the history to be equal to the number of epochs"
