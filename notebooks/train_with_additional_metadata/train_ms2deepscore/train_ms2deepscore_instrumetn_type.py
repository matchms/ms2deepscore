import os
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2deepscore_wrapper
from ms2deepscore.utils import load_pickled_file
from ms2deepscore.MetadataFeatureGenerator import StandardScaler, CategoricalToBinary, OneHotEncoder


if __name__ == "__main__":
    data_directory = "../../../../data/"
    ionisation_mode = "both"

    instrument_type_categories = {'Orbitrap': ['ESI-Orbitrap', 'LC-ESI-Orbitrap', 'LC-ESI-HCD; Velos', 'LC-ESI-CID; Velos', 'LC-ESI-Q-Exactive Plus', 'LC-ESI-CID; Lumos', 'LC-ESI-HCD; Lumos', '-Q-Exactive Plus Orbitrap Res 70k', '-Q-Exactive Plus Orbitrap Res 14k', 'LC-ESI-Q-Exactive Plus Orbitrap Res 70k', 'DI-ESI-Orbitrap', 'LC-ESI-Q-Exactive Plus Orbitrap Res 14k', 'ESI-HCD', 'DI-ESI-Q-Exactive Plus', 'APCI-Orbitrap', 'DI-ESI-Q-Exactive'],
              'TOF': ['ESI-qTof', 'LC-ESI-qTof', 'LC-ESI-Maxis II HD Q-TOF Bruker', 'ESI-LC-ESI-QTOF', 'N/A-ESI-QTOF', 'ESI-qToF', 'ESI-LC-Q-TOF/MS', 'LC-ESI- impact HD', 'DI-ESI-qTof', 'LC-ESI-qToF', 'LC-ESI-qTOF', '-Maxis HD qTOF', 'LC-ESI-Maxis HD qTOF', 'ESI-Q-TOF', 'ESI-qTOF', 'ESI-LC-ESI-ITTOF', 'LC-APCI-qTof', 'LC-ESIMS-qTOF', 'ESI-UPLC-ESI-QTOF', 'LC-ESI-QTOF-LC-ESI-QTOF', 'APCI-qTof', 'ESI-HPLC-ESI-TOF', 'LC-ESI-ITTOF-LC-ESI-ITTOF'],
              'Fourier Transform': ['N/A-ESI-QFT', 'ESI-Hybrid FT', 'ESI-LC-ESI-ITFT', 'LC-ESI-ITFT-LC-ESI-ITFT', 'ESI-LC-ESI-QFT', 'DI-ESI-Hybrid FT', 'LC-ESI-Hybrid FT', 'ESI-APCI-ITFT', 'ESI-ESI-ITFT', 'LC-ESI-LTQ-FTICR', 'LC-ESI-Hybrid Ft', 'ESI-IT-FT/ion trap with FTMS', 'ESI-ESI-FTICR', 'ESI-IT-FT/ion trap with FTMS'],
              'Quadrupole': ['ESI-QQQ', 'ESI-Flow-injection QqQ/MS', 'ESI-LC-ESI-QQ', 'Positive-Quattro_QQQ:40eV', 'Positive-Quattro_QQQ:25eV', 'Positive-Quattro_QQQ:10eV', 'ESI-LC-APPI-QQ', 'Negative-Quattro_QQQ:40eV', 'Negative-Quattro_QQQ:25eV', 'Negative-Quattro_QQQ:10eV', 'LC-ESI-QQQ', 'LC-ESI-QQ-LC-ESI-QQ', 'ESI-QqQ', 'FAB-BEqQ/magnetic and electric sectors with quadrupole', 'APCI-QQQ', 'DI-ESI-QQQ', 'EI-QQQ', 'in source ESI-QqQ', 'ESI-LC-ESI-Q'],
              'Ion Trap': ['N/A-Linear Ion Trap', 'LC-ESI-Ion Trap', 'ESI-Ion Trap', 'ESI-LC-ESI-IT', 'DI-ESI-Ion Trap', 'ESI-IT/ion trap', 'APCI-Ion Trap', 'ESI or APCI-IT/ion trap', 'CI (MeOH)-IT/ion trap', 'ESI-QIT', 'DIRECT INFUSION NANOESI-ION TRAP-DIRECT INFUSION NANOESI-ION TRAP', 'CI-IT/ion trap']}

    additional_metadata = (StandardScaler("precursor_mz", 0, 1000),
                           CategoricalToBinary("ionmode", "positive", "negative"),
                           OneHotEncoder("source_instrument", instrument_type_categories["Orbitrap"]),
                           OneHotEncoder("source_instrument", instrument_type_categories['TOF']),
                           OneHotEncoder("source_instrument", instrument_type_categories['Fourier Transform']),
                           OneHotEncoder("source_instrument", instrument_type_categories['Quadrupole']),
                           OneHotEncoder("source_instrument", instrument_type_categories['Ion Trap']))

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
