import os
import shutil
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2deepscore.utils import load_pickled_file
from ms2deepscore.MetadataFeatureGenerator import StandardScaler
from ms2deepscore.train_new_model.visualize_results import create_all_plots

if __name__ == "__main__":
    data_directory = "../../../../data/"
    ionisation_mode = "both"

    additional_metadata = (StandardScaler("precursor_mz", 0, 1000),
                           )
    binning_file_label = ""
    for metadata_generator in additional_metadata:
        binning_file_label += metadata_generator.metadata_field + "_"
    base_dims = (2000, 2000)

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
    training_spectra = training_spectra_positive + training_spectra_negative
    validation_spectra = validation_spectra_positive+validation_spectra_negative

    tanimoto_scores_file_name = os.path.join(data_directory, "tanimoto_scores",
                                             f"all_tanimoto_scores.pickle")
    train_ms2deepscore_wrapper(training_spectra, validation_spectra, model_folder_file_name,
                               base_dims=base_dims, additional_metadata=additional_metadata,
                               tanimoto_scores_file_name=tanimoto_scores_file_name)
    # copy the current script to that folder for backup reference
    shutil.copyfile(os.path.abspath(__file__), os.path.join(model_folder_file_name, "training_script.py"))
    create_all_plots(model_folder_file_name)
