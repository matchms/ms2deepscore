import pytest
import numpy as np
from matchms import Spectrum
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel, train
from ms2deepscore.MetadataFeatureGenerator import (MetadataVectorizer,
                                                   StandardScaler)
from ms2deepscore.data_generators import DataGeneratorPytorch, tensorize_spectra
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper
from ms2deepscore.data_generators import TensorizationSettings


@pytest.fixture
def dummy_spectra():
    spectrum1 = Spectrum(mz=np.array([101, 202, 303.0]),
                         intensities=np.array([0.1, 0.2, 1.0]),
                         metadata={"precursor_mz": 222.2}
                         )
    spectrum2 = Spectrum(mz=np.array([101.5, 202.5, 303.0]),
                         intensities=np.array([0.1, 0.2, 1.0]),
                         metadata={"precursor_mz": 333.3})
    return [spectrum1, spectrum2]


@pytest.fixture
def simple_training_spectra():
    """Creates many random versions of two very differntly looking types of spectra.
    They come with very different compound annotations so that a model should easily be able to learn those.
    """
    spectra = []
    for _ in range(1000):
        spectra.append(
            Spectrum(mz=np.sort(np.random.uniform(0, 100, 10)),
                     intensities=np.random.uniform(0.2, 1, 10),
                     metadata={
                         "precursor_mz": 222.2,
                         "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                         "inchi": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
                         "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                     },
                     )
        )
        spectra.append(
            Spectrum(mz=np.sort(np.random.uniform(100, 200, 10)),
                     intensities=np.random.uniform(0.2, 1, 10),
                     metadata={
                         "precursor_mz": 444.4,
                         "smiles": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@@H](C(=O)O)N)OC(=O)CCCCCCCCCCCCCCCCC",
                         "inchi": "InChI=1S/C42H82NO10P/c1-3-5-7-9-11-13-15-17-19-21-23-25-27-29-31-33-40(44)50-35-38(36-51-54(48,49)52-37-39(43)42(46)47)53-41(45)34-32-30-28-26-24-22-20-18-16-14-12-10-8-6-4-2/h38-39H,3-37,43H2,1-2H3,(H,46,47)(H,48,49)/t38-,39+/m1/s1",
                         "inchikey": "TZCPCKNHXULUIY-RGULYWFUSA-N",
                     },
                     )
        )
    return spectra    


def test_siamese_model_defaults():
    tensorization_settings = TensorizationSettings(min_mz=10,
                                                   max_mz=1000,
                                                   mz_bin_width=0.1,
                                                   intensity_scaling=0.5,
                                                   additional_metadata=())    # Create the model instance
    model = SiameseSpectralModel(tensorization_settings)

    assert model.model_parameters == {
        'base_dims': (1000, 800, 800),
        'embedding_dim': 400,
        'train_binning_layer': True,
        'group_size': 30,
        'output_per_group': 3,
        'dropout_rate': 0.2,
        'peak_inputs': 9900,
        'additional_inputs': 0
    }


def test_siamese_model_forward_pass(dummy_spectra):
    tensorization_settings = TensorizationSettings(mz_bin_width=1,)
    model = SiameseSpectralModel(tensorization_settings)
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, tensorization_settings)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert similarity_score.shape[0] == 2


def test_siamese_model_no_binning_layer(dummy_spectra):
    tensorization_settings = TensorizationSettings(mz_bin_width=1,)
    model = SiameseSpectralModel(tensorization_settings, train_binning_layer=False)
    assert not model.model_parameters["train_binning_layer"]

    # Test forward pass
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, tensorization_settings)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert similarity_score.shape[0] == 2


def test_siamese_model_additional_metadata(dummy_spectra):
    tensorization_settings = TensorizationSettings(
        mz_bin_width=0.1,
        additional_metadata=[("StandardScaler", {"metadata_field": "precursor_mz",
                                                 "mean": 200.0,
                                                 "standard_deviation": 250.0}), ])

    model = SiameseSpectralModel(tensorization_settings, train_binning_layer=False)

    # Test forward pass
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, tensorization_settings)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert similarity_score.shape[0] == 2
    assert model.encoder.dense_layers[0].weight.shape[1] == 9901

    # Include dense binning layer
    model = SiameseSpectralModel(tensorization_settings)

    # Test forward pass
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, tensorization_settings)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert model.encoder.dense_layers[0].weight.shape[1] == 991

    tensorisaton_settings = TensorizationSettings(mz_bin_width=0.1, )
    # Compare to no additional_metadata
    model = SiameseSpectralModel(tensorisaton_settings)

    # Test forward pass
    spec_tensors, meta_tensors = tensorize_spectra(dummy_spectra, tensorisaton_settings)
    similarity_score = model(spec_tensors, spec_tensors, meta_tensors, meta_tensors)
    assert model.encoder.dense_layers[0].weight.shape[1] == 990


def test_model_training(simple_training_spectra):
    # Select pairs
    settings = SettingsMS2Deepscore({
        "tanimoto_bins": np.array([(0, 0.5), (0.5, 1)]),
        "average_pairs_per_bin": 20
    })
    scp_simple, _ = select_compound_pairs_wrapper(simple_training_spectra, settings)
    tensorization_settings = TensorizationSettings(min_mz=0, max_mz=200, mz_bin_width=0.2,
                                                   intensity_scaling=0.5,)
    # Create generators
    train_generator_simple = DataGeneratorPytorch(
        spectrums=simple_training_spectra,
        tensorization_settings=tensorization_settings,
        selected_compound_pairs=scp_simple,
        batch_size=2,
        num_turns=20,
    )

    val_generator_simple = DataGeneratorPytorch(
        spectrums=simple_training_spectra,
        tensorization_settings=tensorization_settings,
        selected_compound_pairs=scp_simple,
        batch_size=2,
        num_turns=2,
        use_fixed_set=True,
    )

    # Create and train model
    model_simple = SiameseSpectralModel(tensorization_settings, train_binning_layer=False)
    losses, val_losses, collection_targets = train(model_simple, train_generator_simple,
                                                   val_generator=val_generator_simple,
                                                   num_epochs=25,
                                                   learning_rate=0.001, lambda_l1=0, lambda_l2=0,
                                                   progress_bar=False, early_stopping=False,
                                                   )

    assert len(losses) == len(val_losses) == 25
    # Check if model trained to at least an OK result
    assert np.mean(losses[-5:]) < 0.03, "Training was not succesfull!"
    # Check if bias in data is handled correctly
    assert (np.array(collection_targets) == 1).sum() == 500
    assert (np.array(collection_targets) < .2).sum() == 500
