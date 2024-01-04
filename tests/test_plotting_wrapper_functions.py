import numpy as np
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import \
    create_all_plots
from tests.create_test_spectra import pesticides_test_spectra


def test_benchmark_wrapper(tmp_path):
    spectra = pesticides_test_spectra()
    create_all_plots(predictions=np.random.random((len(spectra), len(spectra))) ** 2,
                     true_values=np.random.random((len(spectra), len(spectra))) ** 2,
                     val_spectra_1=spectra,
                     val_spectra_2=spectra,
                     benchmarking_results_folder=tmp_path,
                     file_name_prefix="negative_negative")


def test_benchmark_wrapper_not_symmetric(tmp_path):
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:20]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[20:]]

    create_all_plots(predictions=np.random.random((len(positive_mode_spectra), len(negative_mode_spectra))) ** 2,
                     true_values=np.random.random((len(positive_mode_spectra), len(negative_mode_spectra))) ** 2,
                     val_spectra_1=positive_mode_spectra,
                     val_spectra_2=negative_mode_spectra,
                     benchmarking_results_folder=tmp_path,
                     file_name_prefix="positive_negative")
