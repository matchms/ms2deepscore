"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""

import os
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from matchms import Spectrum
from matplotlib import pyplot as plt
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorCherrypicked
from ms2deepscore.models import SiameseModel
from ms2deepscore.train_new_model.spectrum_pair_selection import select_compound_pairs_wrapper
from ms2deepscore.utils import (create_dir_if_missing,
                                return_non_existing_file_name,
                                save_pickled_file)


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
        The folder that will save the spectra if provided."""

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
        create_dir_if_missing(save_folder)
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
        additional_metadata,
        results_folder,
        epochs=150,
        base_dims=(500, 500),
        embedding_dim=200,
        average_pairs_per_bin=20,
        max_pairs_per_bin=100):
    """Full workflow to train a MS2DeepScore model.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    # Set file names and create dirs
    create_dir_if_missing(results_folder)
    output_model_file_name = return_non_existing_file_name(os.path.join(results_folder, "ms2deepscore_model.hdf5"))
    ms2ds_history_file_name = return_non_existing_file_name(os.path.join(results_folder, "history.txt"))
    ms2ds_history_plot_file_name = return_non_existing_file_name(os.path.join(results_folder, "history.svg"))
    binned_spectra_folder = os.path.join(results_folder, "binned_spectra")
    create_dir_if_missing(binned_spectra_folder)

    selected_compound_pairs_training, selected_training_spectra = select_compound_pairs_wrapper(training_spectra,
                                                                                                average_pairs_per_bin=average_pairs_per_bin,
                                                                                                max_pairs_per_bin=max_pairs_per_bin)
    selected_compound_pair_val, selected_validation_spectra = select_compound_pairs_wrapper(validation_spectra,
                                                                                            average_pairs_per_bin=average_pairs_per_bin,
                                                                                            max_pairs_per_bin=max_pairs_per_bin)

    # Created binned spectra.
    binned_spectrums_training, binned_spectrums_val, spectrum_binner = \
        bin_spectra(selected_training_spectra, selected_validation_spectra, additional_metadata, binned_spectra_folder)

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))

    training_generator = DataGeneratorCherrypicked(
        binned_spectrums_training,
        selected_inchikeys=list(
            {s.get("inchikey")[:14] for s in binned_spectrums_training}
        ),
        selected_compound_pairs=selected_compound_pairs_training,
        spectrum_binner=spectrum_binner,
        same_prob_bins=same_prob_bins,
        num_turns=2,
        augment_noise_max=10,
        augment_noise_intensity=0.01,
    )

    validation_generator = DataGeneratorCherrypicked(
        binned_spectrums_val,
        selected_inchikeys=list({s.get("inchikey")[:14] for s in binned_spectrums_val}),
        selected_compound_pairs=selected_compound_pair_val,
        spectrum_binner=spectrum_binner,
        same_prob_bins=same_prob_bins,
        num_turns=10,  # Number of pairs for each InChiKey14 during each epoch.
        # To prevent data augmentation
        augment_removal_max=0,
        augment_removal_intensity=0,
        augment_intensity=0,
        augment_noise_max=0,
        use_fixed_set=True,
    )

    model = SiameseModel(
        spectrum_binner,
        base_dims=base_dims,
        embedding_dim=embedding_dim,
        dropout_rate=0.2,
    )

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError()],
    )
    # Save best model and include early stopping
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_model_file_name,
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )
    earlystopper_scoring_net = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=30, verbose=1
    )
    # Fit model and save history
    history = model.model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpointer, earlystopper_scoring_net],
    )
    model.load_weights(output_model_file_name)
    model.save(output_model_file_name)

    # Save history
    with open(ms2ds_history_file_name, "w", encoding="utf-8") as f:
        f.write(str(history))
    # Save plot of history
    plot_history(history.history, ms2ds_history_plot_file_name)


def plot_history(history: Dict[str, List[float]], file_name: Optional[str] = None):
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()
