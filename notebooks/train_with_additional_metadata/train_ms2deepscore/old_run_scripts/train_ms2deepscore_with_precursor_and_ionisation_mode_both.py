import os
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2deepscore.utils import load_pickled_file
from ms2deepscore.MetadataFeatureGenerator import StandardScaler, CategoricalToBinary


if __name__ == "__main__":
    data_directory = "../../../../data/"
    ionisation_mode = "both"

    additional_metadata = (StandardScaler("precursor_mz", 0, 1000),
                           CategoricalToBinary("ionmode", "positive", "negative"))
    binning_file_label = ""
    for metadata_generator in additional_metadata:
        binning_file_label += metadata_generator.metadata_field + "_"
    print(binning_file_label)
    base_dims = (500, 500)

    # Define a neural net structure label
    neural_net_structure_label = ""
    for layer in base_dims:
        neural_net_structure_label += str(layer) + "_"
    neural_net_structure_label += "layers"

    model_folder_file_name = os.path.join(data_directory, "trained_models",
                 f"ms2deepscore_model_{ionisation_mode}_mode_{binning_file_label}{neural_net_structure_label}")
    print(f"The model will be stored in the folder: {model_folder_file_name}")

    training_spectra_positive = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                      f"positive_training_spectra.pickle"))
    validation_spectra_positive = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                        f"positive_validation_spectra.pickle"))
    training_spectra_negative = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                      f"negative_training_spectra.pickle"))
    validation_spectra_negative = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                        f"negative_validation_spectra.pickle"))
    training_spectra = training_spectra_positive+training_spectra_negative
    validation_spectra = validation_spectra_positive+validation_spectra_negative

    train_ms2deepscore_wrapper(training_spectra, validation_spectra, model_folder_file_name,
                               base_dims=base_dims, additional_metadata=additional_metadata)
