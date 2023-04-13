import pickle
import os
from ms2deepscore.MetadataFeatureGenerator import CategoricalToBinary, StandardScaler


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object

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


def get_spectra(ion_mode):
    # Load in Spectra
    training_spectra = load_pickled_file(os.path.join(
        data_directory, "training_and_validation_split",
        f"{ion_mode}_training_spectra.pickle"))
    validation_spectra = load_pickled_file(os.path.join(
        data_directory, "training_and_validation_split",
        f"{ion_mode}_validation_spectra.pickle"))
    return training_spectra, validation_spectra

def bin_spectra_both_ionmodes(data_directory,
                additional_input,
                file_label,):

    ion_mode = "both_ionmodes"
    # Check if the current notebook has the right name
    current_file_name = os.path.basename(__file__)
    assert current_file_name == "bin_spectra_" + file_label + ".py"
    training_spectra, validation_spectra = get_spectra("positive")
    training_spectra, validation_spectra = get_spectra("negative")

    # Loading in tensorflow can take long therfore the import is done here
    from ms2deepscore.SpectrumBinner import SpectrumBinner

    # Spectrum binning
    spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0,
                                     peak_scaling=0.5,
                                     additional_metadata=additional_input)
    binned_training_spectrums = spectrum_binner.fit_transform(training_spectra)
    binned_validation_spectrums = spectrum_binner.transform(validation_spectra)

    # Save the results
    spectrum_binner_file_name = return_non_existing_file_name(os.path.join(data_directory, "binned_spectra",
                                                                           f"spectrum_binner_{file_label}_{ion_mode}.pickle"))
    binned_training_spectra_file_name = return_non_existing_file_name(os.path.join(data_directory, "binned_spectra",
                                                                                   f"binned_training_spectra{file_label}_{ion_mode}.pickle"))
    binned_validation_spectra_file_name = return_non_existing_file_name(os.path.join(data_directory, "binned_spectra",
                                                                                     f"binned_validation_spectra{file_label}_{ion_mode}.pickle"))

    save_pickled_file(spectrum_binner, spectrum_binner_file_name)
    save_pickled_file(binned_training_spectrums, binned_training_spectra_file_name)
    save_pickled_file(binned_validation_spectrums, binned_validation_spectra_file_name)


def bin_spectra_per_ionmode(data_directory,
                additional_input,
                file_label,
                ion_mode):
    # Check if the current notebook has the right name
    current_file_name = os.path.basename(__file__)
    assert current_file_name == "bin_spectra_" + file_label + ".py"

    # Load in Spectra
    training_spectra = load_pickled_file(os.path.join(
        data_directory, "training_and_validation_split",
        f"{ion_mode}_training_spectra.pickle"))
    validation_spectra = load_pickled_file(os.path.join(
        data_directory, "training_and_validation_split",
        f"{ion_mode}_validation_spectra.pickle"))

    # Loading in tensorflow can take long therfore the import is done here
    from ms2deepscore.SpectrumBinner import SpectrumBinner

    # Spectrum binning
    spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0,
                                     peak_scaling=0.5,
                                     additional_metadata=additional_input)
    binned_training_spectrums = spectrum_binner.fit_transform(training_spectra)
    binned_validation_spectrums = spectrum_binner.transform(validation_spectra)

    # Save the results
    spectrum_binner_file_name = return_non_existing_file_name(os.path.join(data_directory, "binned_spectra",
                                             f"spectrum_binner_{file_label}_{ion_mode}.pickle"))
    binned_training_spectra_file_name = return_non_existing_file_name(os.path.join(data_directory, "binned_spectra",
                                                     f"binned_training_spectra{file_label}_{ion_mode}.pickle"))
    binned_validation_spectra_file_name = return_non_existing_file_name(os.path.join(data_directory, "binned_spectra",
                                                       f"binned_validation_spectra{file_label}_{ion_mode}.pickle"))

    save_pickled_file(spectrum_binner, spectrum_binner_file_name)
    save_pickled_file(binned_training_spectrums, binned_training_spectra_file_name)
    save_pickled_file(binned_validation_spectrums, binned_validation_spectra_file_name)


if __name__ == "__main__":
    data_directory = "../../../../data/"
    # Define MetadataFeature generators.
    additional_input = ()
    file_label = "default"
    bin_spectra(data_directory, additional_input, file_label)