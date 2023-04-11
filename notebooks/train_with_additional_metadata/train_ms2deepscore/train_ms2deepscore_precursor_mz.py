import pickle
import os
import numpy as np
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from tensorflow.keras.callbacks import (  # pylint: disable=import-error
    EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error
import tensorflow as tf

data_directory = "../../../data/"
binning_file_label = "precursor_mz"
neural_net_structure_label = "_2000_2000_layers"

def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def train_ms2ds_model(binned_spectrums_training,
                      binned_spectrums_val,
                      spectrum_binner,
                      tanimoto_df,
                      output_model_file_name,
                      epochs=150):
    assert not os.path.isfile(output_model_file_name), "The MS2Deepscore output model file name already exists"

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))

    training_generator = DataGeneratorAllInchikeys(
        binned_spectrums_training,
        selected_inchikeys=list({s.get("inchikey")[:14] for s in binned_spectrums_training}),
        reference_scores_df=tanimoto_df,
        dim=len(spectrum_binner.known_bins), # The number of bins created
        same_prob_bins=same_prob_bins,
        num_turns=2,
        augment_vnoise_max=10,
        augment_noise_intensity=0.01,
        additional_input=spectrum_binner.additional_metadata)

    validation_generator = DataGeneratorAllInchikeys(
        binned_spectrums_val,
        selected_inchikeys=list({s.get("inchikey")[:14] for s in binned_spectrums_val}),
        reference_scores_df=tanimoto_df,
        dim=len(spectrum_binner.known_bins),  # The number of bins created
        same_prob_bins=same_prob_bins,
        num_turns=10, # Number of pairs for each InChiKey14 during each epoch.
        # To prevent data augmentation
        augment_removal_max=0, augment_removal_intensity=0, augment_intensity=0, augment_noise_max=0, use_fixed_set=True,
        additional_input=spectrum_binner.additional_metadata

    )

    model = SiameseModel(spectrum_binner, base_dims=(2000, 2000), embedding_dim=200, dropout_rate=0.2)

    model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

    # Save best model and include early stopping
    checkpointer = ModelCheckpoint(filepath=output_model_file_name, monitor='val_loss', mode="min", verbose=1, save_best_only=True)
    earlystopper_scoring_net = EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1)
    # Fit model and save history
    history = model.model.fit(training_generator, validation_data=validation_generator, epochs=epochs, verbose=1,
                              callbacks=[checkpointer, earlystopper_scoring_net])
    model.load_weights(output_model_file_name)
    model.save(output_model_file_name)
    return history.history


spectrum_binner = load_pickled_file(os.path.join(data_directory, "binned_spectra", "spectrum_binner_" + binning_file_label + ".pickle"))
binned_training_spectra = load_pickled_file(os.path.join(data_directory, "binned_spectra", "binned_training_spectra_" + binning_file_label + ".pickle"))
binned_validation_spectra = load_pickled_file(os.path.join(data_directory, "binned_spectra", "binned_validation_spectra_" + binning_file_label + ".pickle"))

tanimoto_scores = load_pickled_file(os.path.join(data_directory, "tanimoto_scores.pickle"))

model_file_name = os.path.join(data_directory, "trained_models", f"ms2deepscore_model_{binning_file_label}{neural_net_structure_label}.pickle")
history_file_name = os.path.join(data_directory, "trained_models", f"history_{binning_file_label}{neural_net_structure_label}.svg")
history = train_ms2ds_model(binned_training_spectra, binned_validation_spectra, spectrum_binner,
                            tanimoto_scores,
                            model_file_name)

from typing import List, Dict, Optional
from matplotlib import pyplot as plt

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

plot_history(history, history_file_name)
