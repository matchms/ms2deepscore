"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""

import os
from typing import Optional
from matplotlib import pyplot as plt
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      train)
from ms2deepscore.SettingsMS2Deepscore import (GeneratorSettings,
                                               SettingsMS2Deepscore,
                                               TensorizationSettings)
from ms2deepscore.train_new_model.data_generators import DataGeneratorPytorch
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper
from ms2deepscore.train_new_model.ValidationLossCalculator import \
    ValidationLossCalculator


def train_ms2ds_model(
        training_spectra,
        validation_spectra,
        results_folder,
        model_settings: SettingsMS2Deepscore,
        generator_settings: GeneratorSettings,
        ):
    """Full workflow to train a MS2DeepScore model.
    """
    model_directory = os.path.join(results_folder, model_settings.model_directory_name)
    os.makedirs(model_directory, exist_ok=True)
    # Save settings
    model_settings.save_to_file(os.path.join(model_directory, "settings.json"))

    output_model_file_name = os.path.join(model_directory, model_settings.model_file_name)
    ms2ds_history_plot_file_name = os.path.join(model_directory, model_settings.history_plot_file_name)

    selected_compound_pairs_training, selected_training_spectra = select_compound_pairs_wrapper(
        training_spectra, settings=generator_settings)

    tensoriztion_settings = TensorizationSettings()
    # Create generators
    train_generator = DataGeneratorPytorch(
        spectrums=selected_training_spectra,
        tensorization_settings=tensoriztion_settings,
        selected_compound_pairs=selected_compound_pairs_training,
        generator_settings=generator_settings
    )

    model = SiameseSpectralModel(tensorisaton_settings=tensoriztion_settings,
                                 base_dims=model_settings.base_dims,
                                 embedding_dim=model_settings.embedding_dim,
                                 dropout_rate=model_settings.dropout_rate,
                                 train_binning_layer= model_settings.train_binning_layer,
                                 group_size = model_settings.train_binning_layer_group_size,
                                 output_per_group = model_settings.train_binning_layer_output_per_group,
                                 )

    validation_loss_calculator = ValidationLossCalculator(validation_spectra,
                                                          score_bins=generator_settings.same_prob_bins)

    history = train(model,
                    train_generator,
                    num_epochs=model_settings.epochs,
                    learning_rate=model_settings.learning_rate,
                    validation_loss_calculator=validation_loss_calculator,
                    patience=model_settings.patience,
                    loss_function=model_settings.loss_function,
                    checkpoint_filename=output_model_file_name, lambda_l1=0, lambda_l2=0)
    # Save plot of history
    plot_history(history["losses"], history["val_losses"], ms2ds_history_plot_file_name)


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
