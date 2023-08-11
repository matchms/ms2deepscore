"""A script that trains a MS2Deepscore model with default settings
This script is not needed for normally running MS2Deepscore, it is only needed to to train new models
"""

import os
from os import PathLike
from typing import Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
from matchms import Spectrum
from matplotlib import pyplot as plt
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from ms2deepscore.train_new_model.calculate_tanimoto_matrix import \
    calculate_tanimoto_scores_unique_inchikey
from ms2deepscore.utils import (load_pickled_file,
                                return_non_existing_file_name,
                                save_pickled_file)


def bin_spectra(
    training_spectra: List[Spectrum],
    validation_spectra: List[Spectrum],
    additional_metadata=(),
    save_folder=None,
):
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
        if not os.path.exists(save_folder):
            assert not os.path.isfile(save_folder), "The folder specified is a file"
            os.mkdir(save_folder)
        save_pickled_file(
            binned_spectrums_training,
            return_non_existing_file_name(
                os.path.join(save_folder, "binned_training_spectra.pickle")
            ),
        )
        save_pickled_file(
            binned_spectrums_val,
            return_non_existing_file_name(
                os.path.join(save_folder, "binned_validation_spectra.pickle")
            ),
        )
        save_pickled_file(
            spectrum_binner,
            return_non_existing_file_name(
                os.path.join(save_folder, "spectrum_binner.pickle")
            ),
        )
    return binned_spectrums_training, binned_spectrums_val, spectrum_binner


def train_ms2ds_model(
    binned_spectrums_training,
    binned_spectrums_val,
    spectrum_binner,
    tanimoto_df,
    output_model_file_name,
    epochs=150,
    base_dims=(500, 500),
    embedding_dim=200,
):
    """Full workflow to train a MS2DeepScore model.
    """
    # pylint: disable=too-many-arguments
    assert not os.path.isfile(
        output_model_file_name
    ), "The MS2Deepscore output model file name already exists"

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))

    training_generator = DataGeneratorAllInchikeys(
        binned_spectrums_training,
        selected_inchikeys=list(
            {s.get("inchikey")[:14] for s in binned_spectrums_training}
        ),
        reference_scores_df=tanimoto_df,
        spectrum_binner=spectrum_binner,
        same_prob_bins=same_prob_bins,
        num_turns=2,
        augment_noise_max=10,
        augment_noise_intensity=0.01,
    )

    validation_generator = DataGeneratorAllInchikeys(
        binned_spectrums_val,
        selected_inchikeys=list({s.get("inchikey")[:14] for s in binned_spectrums_val}),
        reference_scores_df=tanimoto_df,
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
    return history.history


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


def train_ms2deepscore_wrapper(
    training_spectra: List[Spectrum],
    validation_spectra: List[Spectrum],
    output_folder: Union[str, PathLike],
    binned_spectrum_folder=None,
    tanimoto_scores_file_name=None,
    epochs: int = 150,
    base_dims=(500, 500),
    additional_metadata=(),
    embedding_dim=200,
):
    """Trains a MS2Deepscore model

    :param training_spectra: The spectra used for training
    :param validation_spectra: The spectra used for validation
    :param output_folder: The folder in which the model and intermediate files should be stored
    :param epochs: The number of epochs used for training
    :param binned_spectrum_folder: The folder in which precalculated embeddings are stored.
    If set to None, they will be calculated.
    :param tanimoto_scores_file_name: The file location of precalculated tanimoto scores.
    If None, these will be calculated.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # creates a folder if it does not yet exist.
    if not os.path.exists(output_folder):
        assert not os.path.isfile(output_folder), "The folder specified is a file"
        os.mkdir(output_folder)

    # Load in tanimoto scores, or calculate tanimoto scores
    if tanimoto_scores_file_name:
        tanimoto_score_df = load_pickled_file(tanimoto_scores_file_name)
    else:
        all_spectra = training_spectra + validation_spectra
        tanimoto_score_df = calculate_tanimoto_scores_unique_inchikey(
            all_spectra, all_spectra
        )

    if binned_spectrum_folder:
        binned_spectrums_training = load_pickled_file(
            os.path.join(binned_spectrum_folder, "binned_training_spectra.pickle")
        )
        binned_spectrums_val = load_pickled_file(
            os.path.join(binned_spectrum_folder, "binned_validation_spectra.pickle")
        )
        spectrum_binner = load_pickled_file(
            os.path.join(binned_spectrum_folder, "spectrum_binner.pickle")
        )
    else:
        binned_spectrums_training, binned_spectrums_val, spectrum_binner = bin_spectra(
            training_spectra,
            validation_spectra,
            additional_metadata=additional_metadata,
            save_folder=output_folder,
        )

    # Train model
    output_model_file_name = return_non_existing_file_name(
        os.path.join(output_folder, "ms2deepscore_model.hdf5")
    )
    history = train_ms2ds_model(
        binned_spectrums_training,
        binned_spectrums_val,
        spectrum_binner,
        tanimoto_score_df,
        output_model_file_name,
        epochs,
        base_dims=base_dims,
        embedding_dim=embedding_dim,
    )

    # Save history
    ms2ds_history_file_name = return_non_existing_file_name(
        os.path.join(output_folder, "history.txt")
    )
    with open(ms2ds_history_file_name, "w", encoding="utf-8") as f:
        f.write(str(history))
    # Save plot of history
    ms2ds_history_plot_file_name = return_non_existing_file_name(
        os.path.join(output_folder, "history.svg")
    )
    plot_history(history, ms2ds_history_plot_file_name)
