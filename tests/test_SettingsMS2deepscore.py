import json

import pytest
import numpy as np
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore


def test_initiate_settingsms2deepscore():
    settings = SettingsMS2Deepscore(**{"epochs": 200, "base_dims": (200, 200)})
    assert settings.epochs == 200
    assert settings.base_dims == (200, 200)


def test_set_unknown_settings():
    """Test if setting unknown parameters raises a ValueError."""
    with pytest.raises(ValueError):
        SettingsMS2Deepscore(**{"test_case": 123})


def test_set_wrong_type_settings():
    """Test if passing wrong types raises a TypeError."""
    with pytest.raises(TypeError):
        SettingsMS2Deepscore(**{"base_dims": 123})  # expects tuple


def test_coerce_list_to_tuple():
    """Test if lists are coerced into tuples where necessary."""
    settings = SettingsMS2Deepscore(**{"base_dims": [200, 200, 200]})
    assert isinstance(settings.base_dims, tuple)
    assert settings.base_dims == (200, 200, 200)


def test_coerce_list_to_numpy_array():
    """Test if lists of lists are converted to np.ndarray."""
    settings = SettingsMS2Deepscore(**{
        "same_prob_bins": [(-0.01, 0.1), (0.1, 0.2), (0.2, 0.9), (0.9, 1.0)]  # Last bin ends at 1
    })
    assert isinstance(settings.same_prob_bins, np.ndarray)
    assert settings.same_prob_bins[-1][1] == 1.0
    assert settings.same_prob_bins.shape == (4, 2)


def test_coerce_string_to_int():
    """Test if strings that represent integers are correctly coerced to int."""
    settings = SettingsMS2Deepscore(**{"epochs": "200"})
    assert isinstance(settings.epochs, int)
    assert settings.epochs == 200


def test_coerce_string_to_float():
    """Test if strings that represent floats are correctly coerced to float."""
    settings = SettingsMS2Deepscore(**{"learning_rate": "0.001"})
    assert isinstance(settings.learning_rate, float)
    assert settings.learning_rate == 0.001


def test_coerce_string_to_bool():
    """Test if strings like 'True', 'False' are coerced to boolean."""
    settings_true = SettingsMS2Deepscore(**{"use_fixed_set": "True"})
    settings_false = SettingsMS2Deepscore(**{"use_fixed_set": "False"})
    assert isinstance(settings_true.use_fixed_set, bool)
    assert settings_true.use_fixed_set is True
    assert isinstance(settings_false.use_fixed_set, bool)
    assert settings_false.use_fixed_set is False


def test_coerce_invalid_string_to_bool():
   """Test that invalid strings for bool raise an exception."""
   with pytest.raises(TypeError):
       SettingsMS2Deepscore(**{"use_fixed_set": "NotAValidBool"})


def test_save_settings(tmp_path):
    """Test saving settings to a file and ensure correct serialization."""
    settings = SettingsMS2Deepscore(epochs=200, base_dims=(200, 200))
    file_name = tmp_path / "settings.json"
    settings.save_to_file(file_name)
    assert file_name.is_file()
    with open(file_name, 'r') as file:
        result = json.load(file)
    assert result["epochs"] == 200
    assert result["base_dims"] == [200, 200]
    assert result["embedding_dim"] == 500


def test_get_dict():
    """Test if the settings dictionary is correctly returned."""
    settings = SettingsMS2Deepscore(epochs=200, base_dims=(200, 200))
    settings_dict = settings.get_dict()
    assert settings_dict["epochs"] == 200
    assert settings_dict["base_dims"] == (200, 200)
    assert settings_dict["embedding_dim"] == 500


def test_get_dict_with_coerced_types():
    """Test if the coerced types (e.g., tuple, Path, etc.) appear correctly in the dictionary."""
    settings = SettingsMS2Deepscore(
        epochs="200",  # string input should be coerced to int
        base_dims=[200, 200],  # list input should be coerced to tuple
        same_prob_bins=[(-0.01, 0.2), (0.2, 1.0)],  # list input should be coerced to np.ndarray
        model_file_name="model.pt",  # string input should be coerced to Path
    )

    settings_dict = settings.get_dict()

    assert settings_dict["epochs"] == 200
    assert settings_dict["base_dims"] == (200, 200)
    assert isinstance(settings_dict["same_prob_bins"], list)  # it will remain a list when serialized
    assert isinstance(settings_dict["same_prob_bins"][0], list)
    assert isinstance(settings_dict["model_file_name"], str)
    assert settings_dict["model_file_name"] == "model.pt"
