# MS2DeepScore model is trained as in the original MS2DeepScore paper.
# Version 0.2.0 is used (like in the paper)
# However, the model is trained on the same training data as the new MS2DeepScore 2.0 model, to ensure fair comparison.
import numpy as np
from matchms.importing import load_from_mgf
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from tqdm import tqdm

from ms2deepscore import SpectrumBinner
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from ms2deepscore.models import SiameseModel
import pandas as pd

training_scores_filepath = "tanimoto_scores_training_spectra_neg.csv"
validation_scores_filepath = "tanimoto_scores_validation_spectra_neg.csv"

training_spectra_file = "/lustre/BIF/nobackup/jonge094/ms2deepscore/data/pytorch/new_corinna_included/training_and_validation_split/negative_training_spectra.mgf"
validation_spectra_file = "/lustre/BIF/nobackup/jonge094/ms2deepscore/data/pytorch/new_corinna_included/training_and_validation_split/negative_validation_spectra.mgf"
save_model_file = "ms2deepscore_model_neg_best.h5"

mz_min=10.0
mz_max=1000.0

validation_tanimoto_scores_df = pd.read_csv(validation_scores_filepath, index_col=0)
unique_validation_inchikeys = set(validation_tanimoto_scores_df.index)
validation_spectra = []
for spectrum in tqdm(load_from_mgf(validation_spectra_file)):
    inchikey = spectrum.get("inchikey")[:14]
    spectrum.set("inchikey", inchikey)
    if np.any((spectrum.mz >= mz_min) & (spectrum.mz <= mz_max)):
        if inchikey in unique_validation_inchikeys:
            validation_spectra.append(spectrum)
        else:
            print(inchikey)
training_tanimoto_scores_df = pd.read_csv(training_scores_filepath, index_col=0)
unique_training_inchikeys = set(training_tanimoto_scores_df.index)

training_spectra = []
for spectrum in tqdm(load_from_mgf(training_spectra_file)):
    inchikey = spectrum.get("inchikey")[:14]
    spectrum.set("inchikey", inchikey)
    if np.any((spectrum.mz >= mz_min) & (spectrum.mz <= mz_max)):
        if inchikey in unique_training_inchikeys:
            training_spectra.append(spectrum)

spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5, allowed_missing_percentage=100)
binned_training_spectra = spectrum_binner.fit_transform(training_spectra)
binned_validation_spectra = spectrum_binner.transform(validation_spectra)

dimension = len(spectrum_binner.known_bins)
data_generator_val = DataGeneratorAllSpectrums(binned_validation_spectra, validation_tanimoto_scores_df,
                                               dim=dimension,
                                               batch_size=32,
                                               num_turns=10,
                                               shuffle=False,
                                               ignore_equal_pairs=True,
                                               same_prob_bins=[(0, 0.1), (0.1, 0.2), (0.2, 0.3),
                                                               (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                                                               (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)],
                                               augment_removal_max=0,
                                               augment_removal_intensity=0,
                                               augment_intensity=0,
                                               augment_noise_max=0,
                                               augment_noise_intensity=0.0,
                                               use_fixed_set=True
                                               )

data_generator_train = DataGeneratorAllSpectrums(binned_training_spectra, training_tanimoto_scores_df,
                                                 dim=dimension,
                                                 batch_size=32,
                                                 num_turns=1,
                                                 shuffle=True,
                                                 ignore_equal_pairs=True,
                                                 same_prob_bins=[(0, 0.1), (0.1, 0.2), (0.2, 0.3),
                                                                 (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                                                                 (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),(0.9, 1.0)],
                                                 augment_removal_max=0.2,
                                                 augment_removal_intensity=0.4,
                                                 augment_intensity=0.4,
                                                 augment_noise_max=10,
                                                 augment_noise_intensity=0.01,
                                                 use_fixed_set=False)


model = SiameseModel(spectrum_binner, base_dims=(500, 500), embedding_dim=200,
                     dropout_rate=0.2,
                     dropout_in_first_layer = False,
                     l1_reg = 1e-6,
                     l2_reg = 1e-6,)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))

checkpointer = ModelCheckpoint(
    filepath = save_model_file,
    monitor='val_loss', mode="min",
    verbose=1,
    save_best_only=True
    )

earlystopper_scoring_net = EarlyStopping(
    monitor='val_loss', mode="min",
    patience=5,
    verbose=1,
    )

model.fit(data_generator_train,
          validation_data=data_generator_val,
          epochs=100,
          callbacks=[
              earlystopper_scoring_net,
              checkpointer,
          ]
          )
