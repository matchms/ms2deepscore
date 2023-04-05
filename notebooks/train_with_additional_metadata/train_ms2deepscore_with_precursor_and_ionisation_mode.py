import pickle
import os
from ms2deepscore.MetadataFeatureGenerator import CategoricalToBinary, StandardScaler
from ms2deepscore.SpectrumBinner import SpectrumBinner

data_directory = "../../../data/"


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object

training_spectra_file_name = os.path.join(data_directory)
validation_spectra_file_name = ""

similarity_scores_file_name = ""
# Load in Spectra
training_spectra = load_pickled_file(training_spectra_file_name)
validation_spectra = load_pickled_file(validation_spectra_file_name)

# Load in similarity scores
similarity_scores = load_pickled_file(similarity_scores_file_name)

# Define MetadataFeature generators.
additional_input = (StandardScaler("precursor_mz", 0, 1000),
                    CategoricalToBinary("ionization_mode", "positive", "negative"))
# Spectrum binning
spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0,
                                 peak_scaling=0.5,
                                 additional_metadata=additional_input)
binned_training_spectrums = spectrum_binner.fit_transform(training_spectra)
binned_validation_spectrums = spectrum_binner.transform(validation_spectra)



# Create a data generator
dimension = len(spectrum_binner.known_bins)
data_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                           dim=dimension, additional_input=additional_input)

# initiate the model
model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200), embedding_dim=200,
                     dropout_rate=0.2, additional_input=len(additional_input))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
model.summary()
keras.utils.plot_model(model.model, show_shapes=True)

# Train the model:
model.fit(data_generator,
          validation_data=data_generator,
          epochs=10)

