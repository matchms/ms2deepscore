import torch

from ms2deepscore.models import load_model


def save_model_for_java(input_model_file_name, output_model_file_name):
    """Saves a model in a file format compatible with java"""
    model = load_model(input_model_file_name)
    nr_of_spectra = 2
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(
        model,
        (torch.rand(nr_of_spectra, model.model_settings.number_of_bins()),
         torch.rand(nr_of_spectra, model.model_settings.number_of_bins()),
         torch.rand(nr_of_spectra, len(model.model_settings.additional_metadata)),
         torch.rand(nr_of_spectra, len(model.model_settings.additional_metadata))))

    # Save the TorchScript model
    traced_script_module.save(output_model_file_name)


def save_model_for_java_output_embeddings(input_model_file_name, output_model_file_name):
    """Saves a model in a file format compatible with java"""
    model = load_model(input_model_file_name)
    nr_of_spectra = 2
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(
        model.encoder,
        (torch.rand(nr_of_spectra, model.model_settings.number_of_bins()),
         torch.rand(nr_of_spectra, len(model.model_settings.additional_metadata)),
         ))

    # Save the TorchScript model
    traced_script_module.save(output_model_file_name)


if __name__ == "__main__":
    # save_model_for_java("../../../data/pytorch/gnps_21_08_23_min_5_at_5_percent/trained_models/both_mode_precursor_mz_ionmode_2000_2000_2000_layers_500_embedding_2024_01_31_11_51_10/ms2deepscore_model.pt",
    #                     "../../../data/pytorch/gnps_21_08_23_min_5_at_5_percent/trained_models/both_mode_precursor_mz_ionmode_2000_2000_2000_layers_500_embedding_2024_01_31_11_51_10/ms2deepscore_java_model.pt")
    save_model_for_java_output_embeddings("../../tests/resources/testmodel.pt", "../../tests/resources/java_ms2deepscore_embedding_model.pt")