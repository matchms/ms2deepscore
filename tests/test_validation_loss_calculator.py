import numpy as np
import pandas as pd
import pytest
from matchms.Spectrum import Spectrum

from ms2deepscore import MS2DeepScore
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import (
    ValidationLossCalculator,
)
from tests.create_test_spectra import (
    pesticides_test_spectra,
    siamese_spectral_model,
)


@pytest.fixture()
def simple_test_spectra():
    rng = np.random.default_rng(42)
    spectra = []
    for i in range(10):
        spectra.append(
            Spectrum(
                mz=np.sort(rng.uniform(0, 100, 10)),
                intensities=rng.uniform(0.2, 1, 10),
                metadata={
                    "precursor_mz": i,
                    "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                    "inchi": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
                    "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                },
            )
        )
        spectra.append(
            Spectrum(
                mz=np.sort(rng.uniform(100, 200, 10)),
                intensities=rng.uniform(0.2, 1, 10),
                metadata={
                    "precursor_mz": i + 10,
                    "smiles": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@@H](C(=O)O)N)OC(=O)CCCCCCCCCCCCCCCCC",
                    "inchi": "InChI=1S/C42H82NO10P/c1-3-5-7-9-11-13-15-17-19-21-23-25-27-29-31-33-40(44)50-35-38(36-51-54(48,49)52-37-39(43)42(46)47)53-41(45)34-32-30-28-26-24-22-20-18-16-14-12-10-8-6-4-2/h38-39H,3-37,43H2,1-2H3,(H,46,47)(H,48,49)/t38-,39+/m1/s1",
                    "inchikey": "TZCPCKNHXULUIY-RGULYWFUSA-N",
                },
            )
        )
    return spectra


@pytest.fixture()
def validation_bins():
    return np.array(
        [
            (0.8, 0.9),
            (0.7, 0.8),
            (0.9, 1.0),
            (0.6, 0.7),
            (0.5, 0.6),
            (0.4, 0.5),
            (0.3, 0.4),
            (0.2, 0.3),
            (0.1, 0.2),
            (-0.01, 0.1),
        ],
        dtype=np.float32,
    )


def test_validation_loss_calculator_smoke(validation_bins):
    model = siamese_spectral_model()
    test_spectra = pesticides_test_spectra()

    validation_loss_calculator = ValidationLossCalculator(
        test_spectra,
        settings=SettingsMS2Deepscore(same_prob_bins=validation_bins),
    )

    # Keep this to stable loss types unless risk_* code path is fixed.
    loss_types = ["mse", "mae", "rmse"]
    losses, binned_losses = validation_loss_calculator.compute_binned_validation_loss(
        model,
        loss_types,
    )

    assert len(losses) == len(loss_types)
    assert len(binned_losses) == len(loss_types)

    for loss_type in loss_types:
        assert loss_type in losses
        assert loss_type in binned_losses
        assert len(binned_losses[loss_type]) == len(validation_bins)
        assert np.all(np.asarray(binned_losses[loss_type]) >= 0)


def test_validation_loss_calculator_stores_unique_tanimoto_matrix(simple_test_spectra, validation_bins):
    validation_loss_calculator = ValidationLossCalculator(
        simple_test_spectra,
        settings=SettingsMS2Deepscore(same_prob_bins=validation_bins),
    )

    unique_inchikeys = sorted({s.get("inchikey")[:14] for s in simple_test_spectra})

    assert validation_loss_calculator.tanimoto_scores.shape == (len(unique_inchikeys), len(unique_inchikeys))
    assert list(validation_loss_calculator.tanimoto_scores.index) == unique_inchikeys
    assert list(validation_loss_calculator.tanimoto_scores.columns) == unique_inchikeys


def test_validation_loss_calculator_chunking_is_close(simple_test_spectra, validation_bins):
    model = siamese_spectral_model()
    settings = SettingsMS2Deepscore(same_prob_bins=validation_bins)

    calculator_large_chunk = ValidationLossCalculator(
        simple_test_spectra,
        settings=settings,
        chunk_size=10_000,
    )
    calculator_small_chunk = ValidationLossCalculator(
        simple_test_spectra,
        settings=settings,
        chunk_size=3,
    )

    loss_types = ["mse", "mae", "rmse"]

    losses_large, binned_large = calculator_large_chunk.compute_binned_validation_loss(model, loss_types)
    losses_small, binned_small = calculator_small_chunk.compute_binned_validation_loss(model, loss_types)

    for loss_type in loss_types:
        assert np.isclose(losses_large[loss_type], losses_small[loss_type], rtol=1e-2, atol=1e-6)
        assert np.allclose(binned_large[loss_type], binned_small[loss_type], rtol=1e-2, atol=1e-6, equal_nan=True)


def test_validation_loss_calculator_rmse_matches_sqrt_of_binned_mse(simple_test_spectra, validation_bins):
    model = siamese_spectral_model()
    settings = SettingsMS2Deepscore(same_prob_bins=validation_bins)

    calculator = ValidationLossCalculator(
        simple_test_spectra,
        settings=settings,
        chunk_size=4,
    )

    losses, binned_losses = calculator.compute_binned_validation_loss(model, ["mse", "rmse"])

    mse_bins = np.asarray(binned_losses["mse"], dtype=float)
    rmse_bins = np.asarray(binned_losses["rmse"], dtype=float)

    assert np.allclose(rmse_bins, np.sqrt(mse_bins), equal_nan=True)
    assert np.isclose(losses["rmse"], np.mean(rmse_bins), equal_nan=True)
    assert np.isclose(losses["mse"], np.mean(mse_bins), equal_nan=True)


def test_validation_loss_calculator_returns_one_value_per_bin(simple_test_spectra, validation_bins):
    model = siamese_spectral_model()
    settings = SettingsMS2Deepscore(same_prob_bins=validation_bins)

    calculator = ValidationLossCalculator(
        simple_test_spectra,
        settings=settings,
        chunk_size=5,
    )

    loss_types = ["mse", "mae", "rmse"]
    losses, binned_losses = calculator.compute_binned_validation_loss(model, loss_types)

    for loss_type in loss_types:
        assert len(binned_losses[loss_type]) == len(validation_bins)
        assert np.isfinite(losses[loss_type]) or np.isnan(losses[loss_type])


def test_validation_loss_calculator_internal_embeddings_shape(simple_test_spectra, validation_bins):
    model = siamese_spectral_model()
    ms2ds_model = MS2DeepScore(model)
    settings = SettingsMS2Deepscore(same_prob_bins=validation_bins)

    calculator = ValidationLossCalculator(
        simple_test_spectra,
        settings=settings,
        chunk_size=5,
    )

    embeddings = calculator._compute_all_embeddings(ms2ds_model)
    assert embeddings.shape[0] == len(simple_test_spectra)
    assert embeddings.ndim == 2


def test_validation_loss_calculator_prediction_block_shape(simple_test_spectra, validation_bins):
    model = siamese_spectral_model()
    ms2ds_model = MS2DeepScore(model)
    settings = SettingsMS2Deepscore(same_prob_bins=validation_bins)

    calculator = ValidationLossCalculator(
        simple_test_spectra,
        settings=settings,
        chunk_size=5,
    )

    embeddings = calculator._compute_all_embeddings(ms2ds_model)
    predictions_df = calculator._compute_prediction_block(embeddings, 0, 4, 4, 8)

    assert isinstance(predictions_df, pd.DataFrame)
    assert predictions_df.shape == (4, 4)
    assert list(predictions_df.index) == [s.get("inchikey")[:14] for s in simple_test_spectra[0:4]]
    assert list(predictions_df.columns) == [s.get("inchikey")[:14] for s in simple_test_spectra[4:8]]


def test_validation_loss_calculator_global_pair_average_shape(simple_test_spectra, validation_bins):
    model = siamese_spectral_model()
    ms2ds_model = MS2DeepScore(model)
    settings = SettingsMS2Deepscore(same_prob_bins=validation_bins)

    calculator = ValidationLossCalculator(
        simple_test_spectra,
        settings=settings,
        chunk_size=4,
    )

    embeddings = calculator._compute_all_embeddings(ms2ds_model)
    average_loss_per_type = calculator._compute_global_average_losses_per_inchikey_pair(
        embeddings,
        ["mse", "mae"],
    )

    unique_inchikeys = sorted({s.get("inchikey")[:14] for s in simple_test_spectra})

    for loss_type in ["mse", "mae"]:
        df = average_loss_per_type[loss_type]
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (len(unique_inchikeys), len(unique_inchikeys))
        assert list(df.index) == unique_inchikeys
        assert list(df.columns) == unique_inchikeys


def test_validation_loss_calculator_chunk_indices():
    ranges = list(ValidationLossCalculator._chunk_indices(10, 3))
    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 10)]


def test_validation_loss_calculator_invalid_loss_type_raises(simple_test_spectra, validation_bins):
    model = siamese_spectral_model()
    settings = SettingsMS2Deepscore(same_prob_bins=validation_bins)

    calculator = ValidationLossCalculator(
        simple_test_spectra,
        settings=settings,
        chunk_size=4,
    )

    with pytest.raises(ValueError, match="not a valid loss type"):
        calculator.compute_binned_validation_loss(model, ["definitely_not_a_loss"])
