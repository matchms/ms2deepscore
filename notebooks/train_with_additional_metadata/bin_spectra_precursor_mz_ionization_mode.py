import pickle
import os
from ms2deepscore.MetadataFeatureGenerator import CategoricalToBinary, StandardScaler

data_directory = "../../data/"
# Define MetadataFeature generators.
additional_input = (StandardScaler("precursor_mz", 0, 1000),
                    CategoricalToBinary("ionization_mode", "positive", "negative"))
file_label = "precursor_mz_ionization_mode"
# Check if the current notebook has the right name
current_file_name = os.path.basename(__file__)
assert current_file_name == "bin_spectra_" + file_label + ".py"

spectrum_binner_file_name = os.path.join(data_directory, "spectrum_binner_" + file_label + ".pickle")
binned_training_spectra_file_name = os.path.join(data_directory, "binned_training_spectra" + file_label + ".pickle")
binned_validation_spectra_file_name = os.path.join(data_directory, "binned_validation_spectra" + file_label + ".pickle")


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


training_spectra_file_name = os.path.join(data_directory, "training_spectra.pickle")
validation_spectra_file_name = os.path.join(data_directory, "validation_spectra.pickle")
similarity_scores_file_name = os.path.join(data_directory, "all_spectra_230201_annotated_min2peaks_tanimoto_scores.pickle")

# Load in Spectra
training_spectra = load_pickled_file(training_spectra_file_name)
validation_spectra = load_pickled_file(validation_spectra_file_name)

# Load in similarity scores
similarity_scores = load_pickled_file(similarity_scores_file_name)
# Loading in tensorflow can take long therfore the import is done here
from ms2deepscore.SpectrumBinner import SpectrumBinner

# Spectrum binning
spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0,
                                 peak_scaling=0.5,
                                 additional_metadata=additional_input)
binned_training_spectrums = spectrum_binner.fit_transform(training_spectra)
binned_validation_spectrums = spectrum_binner.transform(validation_spectra)


def return_non_existing_file_name(file_name):
    """Checks if a path already exists, otherwise creates a new filename with (1)"""
    if not os.path.exists(file_name):
        return file_name
    print(f"The file name already exists: {file_name}")
    file_name_base, file_extension = os.path.splitext(file_name)
    i = 1
    new_file_name = f"{file_name_base}({i}){file_extension}"
    while os.path.exists(new_file_name):
        i += 1
        new_file_name = f"{file_name_base}({i}){file_extension}"
    print(f"Instead the file will be stored in {new_file_name}")
    return new_file_name


def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


save_pickled_file(spectrum_binner, return_non_existing_file_name(spectrum_binner_file_name))
save_pickled_file(binned_training_spectrums, return_non_existing_file_name(binned_training_spectra_file_name))
save_pickled_file(binned_validation_spectrums, return_non_existing_file_name(binned_validation_spectra_file_name))
