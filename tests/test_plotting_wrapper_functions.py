from tests.create_test_spectra import (ms2deepscore_model,
                                       pesticides_test_spectra)

from ms2deepscore.wrapper_functions.plotting_wrapper_functions import (
    BenchmarkingResultsFileNames, benchmark_wrapper)


def test_benchmark_wrapper(tmp_path):
    spectra = pesticides_test_spectra()
    benchmarking_results_file_names = BenchmarkingResultsFileNames(tmp_path, "negative_negative")
    benchmark_wrapper(spectra, spectra, benchmarking_results_file_names, ms2deepscore_model())


def test_benchmark_wrapper_not_symmetric(tmp_path):
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:20]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[20:]]
    benchmarking_results_file_names = BenchmarkingResultsFileNames(tmp_path, "positive_negative")
    benchmark_wrapper(positive_mode_spectra, negative_mode_spectra,
                      benchmarking_results_file_names, ms2deepscore_model())
