import os
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2deepscore.utils import load_pickled_file
from ms2deepscore.MetadataFeatureGenerator import StandardScaler


if __name__ == "__main__":
    data_directory = "../../../../data/"
    ionisation_mode = "positive"

    additional_metadata = (StandardScaler("precursor_mz", 0, 1000),
                           )
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

    training_spectra = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                      f"{ionisation_mode}_training_spectra.pickle"))
    validation_spectra = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                        f"{ionisation_mode}_validation_spectra.pickle"))

    tanimoto_scores_file_name = os.path.join(data_directory, "tanimoto_scores",
                                             f"{ionisation_mode}_tanimoto_scores.pickle")
    train_ms2deepscore_wrapper(training_spectra, validation_spectra, model_folder_file_name,
                               base_dims=base_dims, additional_metadata=additional_metadata, tanimoto_scores_file_name=tanimoto_scores_file_name)
