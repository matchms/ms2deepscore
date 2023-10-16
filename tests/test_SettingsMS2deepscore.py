import pytest
from ms2deepscore.train_new_model.SettingMS2Deepscore import SettingsMS2Deepscore


def test_initiate_settingsms2deepscore():
    settings = SettingsMS2Deepscore({"epochs": 200,
                                     "base_dims": (200, 200)})
    assert settings.epochs == 200
    assert settings.base_dims == (200, 200)
    assert settings.max_pairs_per_bin == 100


def test_set_unknown_settings():
    with pytest.raises(ValueError):
        SettingsMS2Deepscore({"test_case": 123})
