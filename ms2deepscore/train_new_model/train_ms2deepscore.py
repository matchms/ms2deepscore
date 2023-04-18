"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""

import os
from typing import List, Dict, Optional
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matchms import Spectrum
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from ms2deepscore.train_new_model.calculate_tanimoto_matrix import calculate_tanimoto_scores_unique_inchikey


def train_ms2ds_model(training_spectra,
                      validation_spectra,
                      tanimoto_df,
                      output_model_file_name,
                      epochs=150):
    assert not os.path.isfile(output_model_file_name), "The MS2Deepscore output model file name already exists"
    # Bin training spectra
    spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                     allowed_missing_percentage=100.0)
    binned_spectrums_training = spectrum_binner.fit_transform(training_spectra)
    # Bin validation spectra using the binner based on the training spectra.
    # Peaks that do not occur in the training spectra will not be binned in the validaiton spectra.
    binned_spectrums_val = spectrum_binner.transform(validation_spectra)

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))

    training_generator = DataGeneratorAllInchikeys(
        binned_spectrums_training,
        selected_inchikeys=list({s.get("inchikey")[:14] for s in training_spectra}),
        reference_scores_df=tanimoto_df,
        dim=len(spectrum_binner.known_bins), # The number of bins created
        same_prob_bins=same_prob_bins,
        num_turns=2,
        augment_noise_max=10,
        augment_noise_intensity=0.01)

    validation_generator = DataGeneratorAllInchikeys(
        binned_spectrums_val,
        selected_inchikeys=list({s.get("inchikey")[:14] for s in binned_spectrums_val}),
        reference_scores_df=tanimoto_df,
        dim=len(spectrum_binner.known_bins),  # The number of bins created
        same_prob_bins=same_prob_bins,
        num_turns=10, # Number of pairs for each InChiKey14 during each epoch.
        # To prevent data augmentation
        augment_removal_max=0, augment_removal_intensity=0, augment_intensity=0, augment_noise_max=0, use_fixed_set=True
    )

    model = SiameseModel(spectrum_binner, base_dims=(500, 500),
                         embedding_dim=200, dropout_rate=0.2)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])
    # Save best model and include early stopping
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_model_file_name, monitor='val_loss', mode="min",
        verbose=1, save_best_only=True)
    earlystopper_scoring_net = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1)
    # Fit model and save history
    history = model.model.fit(training_generator, validation_data=validation_generator,
                              epochs=epochs, verbose=1,
                              callbacks=[checkpointer, earlystopper_scoring_net])
    model.load_weights(output_model_file_name)
    model.save(output_model_file_name)
    return history.history


def plot_history(history: Dict[str, List[float]],
                 file_name: Optional[str] = None):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()


def train_ms2deepscore_wrapper(training_spectra: List[Spectrum],
                               validation_spectra,
                               output_model_file_name,
                               epochs,
                               ms2ds_history_file_name=None):
    assert not os.path.isfile(output_model_file_name), "The MS2Deepscore output model file name already exists"
    all_spectra = training_spectra + validation_spectra
    tanimoto_score_df = calculate_tanimoto_scores_unique_inchikey(all_spectra, all_spectra)
    history = train_ms2ds_model(training_spectra, validation_spectra,
                                tanimoto_score_df, output_model_file_name,
                                epochs)
    print(f"The training history is: {history}")
    plot_history(history, ms2ds_history_file_name)
