"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""

import os
from typing import Optional
from matplotlib import pyplot as plt
from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      train)
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.data_generators import create_data_generator
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import \
    ValidationLossCalculator


def train_ms2ds_model(
        training_spectra,
        validation_spectra,
        results_folder,
        settings: SettingsMS2Deepscore,
        inchikey_pairs_file: str = None,
        ):
    """Full workflow to train a MS2DeepScore model.
    """
    # Make folder and save settings
    os.makedirs(results_folder, exist_ok=True)
    settings.save_to_file(os.path.join(results_folder, "settings.json"))

    # Create a training generator
    if inchikey_pairs_file is None:
        train_generator = create_data_generator(training_spectra, settings, None)
    else:
        train_generator = create_data_generator(training_spectra, settings,
                                                os.path.join(results_folder, inchikey_pairs_file))
    # Create a validation loss calculator
    validation_loss_calculator = ValidationLossCalculator(validation_spectra,
                                                          settings=settings)

    model = SiameseSpectralModel(settings=settings)

    output_model_file_name = os.path.join(results_folder, settings.model_file_name)

    history = train(model,
                    train_generator,
                    num_epochs=settings.epochs,
                    learning_rate=settings.learning_rate,
                    validation_loss_calculator=validation_loss_calculator,
                    patience=settings.patience,
                    loss_function=settings.loss_function,
                    checkpoint_filename=output_model_file_name, lambda_l1=0, lambda_l2=0)
    return model, history


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
