import json
import os
import pytest
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore, validate_bin_order


def test_initiate_settingsms2deepscore():
    settings = SettingsMS2Deepscore(**{"epochs": 200,
                                       "base_dims": (200, 200)})
    assert settings.epochs == 200
    assert settings.base_dims == (200, 200)


def test_set_unknown_settings():
    with pytest.raises(ValueError):
        SettingsMS2Deepscore(**{"test_case": 123})


def test_set_wrong_type_settings():
    with pytest.raises(TypeError):
        SettingsMS2Deepscore(**{"base_dims": 123})


def test_save_settings(tmp_path):
    settings = SettingsMS2Deepscore(epochs=200,
                                    base_dims=(200,200))
    file_name = os.path.join(tmp_path, "settings.json")
    settings.save_to_file(file_name)
    assert os.path.isfile(file_name)
    with open(file_name, 'r') as file:
        result = json.load(file)
    assert result["epochs"] == 200
    assert result["base_dims"] == [200, 200]
    assert result["embedding_dim"] == 400


def test_get_dict():
    _ = SettingsMS2Deepscore().get_dict()


@pytest.mark.parametrize("bins,correct", [
    ([(-0.01, 1)], True),
    ([(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
      (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (-0.01, 0.1)], True),
    ([(-0.01, 0.6), (0.7, 1.0)], False),  # Test a gap in bins is detected
    ([(0.0, 0.6), (0.7, 1.0)], False),  # Test that the lowest values is below 0.
    ([(-0.3, -0.1), (-0.1, 1.0)], False),  # Test that no bin is entirely below 0.
    ([(0.0, 0.6), (0.6, 0.6), (0.6, 1.0)], False),  # Test no repeating bin borders
    ([(0.0, 0.6), (0.7, 0.6), (0.7, 1.0)], False),  # Test correct order of bin borders
    ([(0.0, 0.5, 1.), (0.5, 0.7, 1.), (0.7, 1.0)], False),  # Test all bins have two elements
])
def test_validate_bin_order(bins, correct):
    if correct:
        validate_bin_order(bins)
    else:
        with pytest.raises(ValueError):
            validate_bin_order(bins)
