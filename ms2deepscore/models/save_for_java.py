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

