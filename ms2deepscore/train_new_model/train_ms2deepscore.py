"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""

import os
from typing import Optional
from matplotlib import pyplot as plt
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      train)
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.data_generators import DataGeneratorPytorch
from ms2deepscore.train_new_model.spectrum_pair_selection import \
    select_compound_pairs_wrapper
from ms2deepscore.train_new_model.ValidationLossCalculator import \
    ValidationLossCalculator


def train_ms2ds_model(
        training_spectra,
        validation_spectra,
        results_folder,
        settings: SettingsMS2Deepscore,
        ):
    """Full workflow to train a MS2DeepScore model.
    """
    os.makedirs(results_folder, exist_ok=True)
    # Save settings
    settings.save_to_file(os.path.join(results_folder, "settings.json"))

    output_model_file_name = os.path.join(results_folder, settings.model_file_name)
    ms2ds_history_plot_file_name = os.path.join(results_folder, settings.history_plot_file_name)

    selected_compound_pairs_training, selected_training_spectra = select_compound_pairs_wrapper(
        training_spectra, settings=settings)

    # Create generators
    train_generator = DataGeneratorPytorch(spectrums=selected_training_spectra,
                                           selected_compound_pairs=selected_compound_pairs_training,
                                           settings=settings)

    model = SiameseSpectralModel(settings=settings)

    validation_loss_calculator = ValidationLossCalculator(validation_spectra,
                                                          settings=settings)

    history = train(model,
                    train_generator,
                    num_epochs=settings.epochs,
                    learning_rate=settings.learning_rate,
                    validation_loss_calculator=validation_loss_calculator,
                    patience=settings.patience,
                    loss_function=settings.loss_function,
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
