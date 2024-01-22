"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""

import os
from typing import Optional
from matplotlib import pyplot as plt
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper
from ms2deepscore.data_generators import DataGeneratorPytorch, TensorizationSettings
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel, train


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

    tensoriztion_settings = TensorizationSettings()
    # todo check if num_turns=2 and augment_noise_max = 10 should still be added.
    # Create generators
    train_generator = DataGeneratorPytorch(
        spectrums=selected_training_spectra,
        tensorization_settings=tensoriztion_settings,
        selected_compound_pairs=selected_compound_pairs_training,
        augment_intensity=0.1,
        batch_size=settings.batch_size,
    )
    selected_compound_pair_val, selected_validation_spectra = select_compound_pairs_wrapper(
        validation_spectra, settings=settings)
    # todo maybe add num_turns=10 # Number of pairs for each InChiKey14 during each epoch. again?
    val_generator = DataGeneratorPytorch(
        spectrums=selected_validation_spectra,
        tensorization_settings=tensoriztion_settings,
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
    model = SiameseSpectralModel(tensorisaton_settings=tensoriztion_settings,
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
