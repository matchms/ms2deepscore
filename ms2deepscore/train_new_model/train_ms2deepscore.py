"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""
import json
import os
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from ms2deepscore.models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      train)
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model import TrainingBatchGenerator, select_compound_pairs_wrapper, SpectrumPairGenerator
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import \
    ValidationLossCalculator


def train_ms2ds_model(
        training_spectra,
        validation_spectra,
        results_folder,
        settings: SettingsMS2Deepscore,
        ):
    """Full workflow to train a MS2DeepScore model.
    """
    # Make folder and save settings
    os.makedirs(results_folder, exist_ok=True)
    settings.save_to_file(os.path.join(results_folder, "settings.json"))

    # Create a training generator
    spectrum_pair_generator = select_compound_pairs_wrapper(training_spectra, settings=settings)
    train_generator = TrainingBatchGenerator(spectrum_pair_generator=spectrum_pair_generator, settings=settings)

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

# def create_data_generator_across_ionmodes(training_spectra,
#                           settings: SettingsMS2Deepscore,
#                           json_save_file=None) -> TrainingBatchGenerator:
#     # todo actually create, both between and across ionmodes.
#     pos_spectra, neg_spectra = split_by_ionmode(training_spectra)
#
#     pos_spectrum_pair_generator = select_compound_pairs_wrapper(pos_spectra, settings=settings)
#     neg_spectrum_pair_generator = select_compound_pairs_wrapper(neg_spectra, settings=settings)
#     pos_neg_spectrum_pair_generator = select_compound_pairs_wrapper_across_ionmode(pos_spectra, neg_spectra, settings)
#
#     if json_save_file is not None:
#         inchikey_pair_generator.save_as_json(json_save_file)
#     # todo possibly create a single TrainingBatchGenerator which takes in 3 generators and pos and neg spectra to iteratively select each one.
#     # Create generators
#     # todo also make sure that the TrainingBatchGenerator can work across ionmodes.
#     train_generator = TrainingBatchGenerator(spectrum_pair_generator=inchikey_pair_generator, settings=settings)
#     return train_generator



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


def save_history(file_name, history):
    def convert_np_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):  # If there are any individual float32 items
            return float(obj)
        raise TypeError("Object not serializable")
    with open(file_name, "w") as file:
        json.dump(history, file, default=convert_np_to_list)
