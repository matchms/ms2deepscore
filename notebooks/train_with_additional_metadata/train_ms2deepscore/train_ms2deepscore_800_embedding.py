import os
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2deepscore.utils import load_pickled_file
from ms2deepscore.MetadataFeatureGenerator import StandardScaler, CategoricalToBinary
from ms2deepscore.train_new_model.visualize_results import create_all_plots


def create_file_name(additional_metadata,
                     base_dims,
                     ionisation_mode,
                     embedding_dims=None):
    binning_file_label = ""
    for metadata_generator in additional_metadata:
        binning_file_label += metadata_generator.metadata_field + "_"

    # Define a neural net structure label
    neural_net_structure_label = ""
    for layer in base_dims:
        neural_net_structure_label += str(layer) + "_"
    neural_net_structure_label += "layers"

    if embedding_dims:
        neural_net_structure_label += f"_{str(embedding_dims)}_embedding"

    model_folder_file_name = f"ms2deepscore_model_{ionisation_mode}_mode_{binning_file_label}{neural_net_structure_label}"
    print(f"The model will be stored in the folder: {model_folder_file_name}")
    return model_folder_file_name


def load_spectra(data_directory, ionisation_mode):
    assert ionisation_mode in ("positive", "negative", "both")
    training_spectra_positive = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                               f"positive_training_spectra.pickle"))
    validation_spectra_positive = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                                 f"positive_validation_spectra.pickle"))
    if ionisation_mode == "positive":
        return training_spectra_positive, validation_spectra_positive

    training_spectra_negative = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                               f"negative_training_spectra.pickle"))
    validation_spectra_negative = load_pickled_file(os.path.join(data_directory, "training_and_validation_split",
                                                                 f"negative_validation_spectra.pickle"))
    if ionisation_mode == "negative":
        return training_spectra_negative, validation_spectra_negative
    training_spectra_both = training_spectra_positive + training_spectra_negative
    validation_spectra_both = validation_spectra_positive + validation_spectra_negative
    if ionisation_mode == "both":
        return training_spectra_both, validation_spectra_both


def train_and_benchmark_wrapper(data_directory,
                                additional_metadata,
                                base_dims,
                                ionisation_mode,
                                embedding_dims=200):
    model_folder_file_name = create_file_name(additional_metadata,
                                              base_dims,
                                              ionisation_mode,
                                              embedding_dims)

    model_folder_file_path = os.path.join(data_directory, "trained_models",
                                          model_folder_file_name)

    training_spectra, validation_spectra = load_spectra(data_directory, ionisation_mode)

    tanimoto_scores_file_name = os.path.join(data_directory, "tanimoto_scores",
                                             f"all_tanimoto_scores.pickle")

    train_ms2deepscore_wrapper(training_spectra, validation_spectra, model_folder_file_path,
                               base_dims=base_dims, additional_metadata=additional_metadata,
                               tanimoto_scores_file_name=tanimoto_scores_file_name,
                               embedding_dim=embedding_dims)
    create_all_plots(model_folder_file_name)

if __name__ == "__main__":
    data_directory = "../../../../data/"

    additional_metadata = (StandardScaler("precursor_mz", 0, 1000),
                           CategoricalToBinary("ionmode", "positive", "negative"),)
    base_dims = (2000, 2000)
    ionisation_mode = "both"
    embedding_dims = 800
    train_and_benchmark_wrapper(data_directory,
                                additional_metadata,
                                base_dims,
                                ionisation_mode,
                                embedding_dims)
