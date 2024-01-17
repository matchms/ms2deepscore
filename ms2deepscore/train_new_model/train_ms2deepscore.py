"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""

import os
from typing import List, Optional
from matchms import Spectrum
from matplotlib import pyplot as plt
from ms2deepscore import SpectrumBinner
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper
from ms2deepscore.utils import return_non_existing_file_name, save_pickled_file
from ms2deepscore.data_generators import DataGeneratorPytorch
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel, train


def bin_spectra(
    training_spectra: List[Spectrum],
    validation_spectra: List[Spectrum],
    additional_metadata=(),
    save_folder=None):
    """Bins spectra and stores binner and binned spectra in the specified folder.
    training_spectra:
        Spectra will be binned and will be used to decide which bins are used.
    validation_spectra:
        These spectra are binned based on the bins determined for the training_spectra.
    additional_metadata:
        Additional metadata that should be used in training the model. e.g. precursor_mz
    save_folder:
        The folder that will save the binned spectra if provided."""

    # Bin training spectra
    spectrum_binner = SpectrumBinner(
        10000,
        mz_min=10.0,
        mz_max=1000.0,
        peak_scaling=0.5,
        allowed_missing_percentage=100.0,
        additional_metadata=additional_metadata,
    )
    binned_spectrums_training = spectrum_binner.fit_transform(training_spectra)
    # Bin validation spectra using the binner based on the training spectra.
    # Peaks that do not occur in the training spectra will not be binned in the validation spectra.
    binned_spectrums_val = spectrum_binner.transform(validation_spectra)

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_pickled_file(binned_spectrums_training, return_non_existing_file_name(
                os.path.join(save_folder, "binned_training_spectra.pickle")), )
        save_pickled_file(binned_spectrums_val, return_non_existing_file_name(
                os.path.join(save_folder, "binned_validation_spectra.pickle")), )
        save_pickled_file(spectrum_binner, return_non_existing_file_name(
                os.path.join(save_folder, "spectrum_binner.pickle")), )
    return binned_spectrums_training, binned_spectrums_val, spectrum_binner


def train_ms2ds_model(
        training_spectra,
        validation_spectra,
        results_folder,
        settings: SettingsMS2Deepscore):
    """Full workflow to train a MS2DeepScore model.
    """
    model_directory = os.path.join(results_folder, settings.model_directory_name)
    os.makedirs(model_directory, exist_ok=True)
    # Save settings
    settings.save_to_file(os.path.join(model_directory, "settings.json"))

    output_model_file_name = os.path.join(model_directory, settings.model_file_name)
    ms2ds_history_plot_file_name = os.path.join(model_directory, settings.history_plot_file_name)

    # todo remove the binned spectra folder and bin spectra in training wrapper

    selected_compound_pairs_training, selected_training_spectra = select_compound_pairs_wrapper(
        training_spectra, settings=settings)

    # todo check if num_turns=2 and augment_noise_max = 10 should still be added.
    # Create generators
    train_generator = DataGeneratorPytorch(
        spectrums=selected_training_spectra,
        min_mz=10, max_mz=1000, mz_bin_width=0.1, intensity_scaling=0.5,
        metadata_vectorizer=None,
        selected_compound_pairs=selected_compound_pairs_training,
        augment_intensity=0.1,
        batch_size=settings.batch_size,
    )
    selected_compound_pair_val, selected_validation_spectra = select_compound_pairs_wrapper(
        validation_spectra, settings=settings)
    # todo maybe add num_turns=10 # Number of pairs for each InChiKey14 during each epoch. again?
    val_generator = DataGeneratorPytorch(
        spectrums=selected_validation_spectra,
        min_mz=10, max_mz=1000, mz_bin_width=0.1, intensity_scaling=0.5,
        metadata_vectorizer=None,
        selected_compound_pairs=selected_compound_pair_val,
        batch_size=settings.batch_size,
        use_fixed_set=True,
        augment_removal_max=0.0,
        augment_removal_intensity=0.0,
        augment_intensity=0.0,
        augment_noise_max=0
    )
    # todo check if dropout rate, layers, base dim and embedding size should still be integrated.
    # todo Check where we specify the loss type. Should that happen here as well, maybe settings?
    model = SiameseSpectralModel(peak_inputs=train_generator.num_bins, additional_inputs=0,
                                 train_binning_layer=False)

    losses, val_losses, collection_targets = train(
        model, train_generator, 200,
        val_generator=val_generator,
        checkpoint_filename=output_model_file_name,
        learning_rate=settings.learning_rate, lambda_l1=0, lambda_l2=0, patience=settings.patience)
    # Save plot of history
    plot_history(losses, val_losses, ms2ds_history_plot_file_name)


def plot_history(losses, val_losses, file_name: Optional[str] = None):
    plt.plot(losses)
    plt.plot(val_losses)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()
