from ms2deepscore.wrapper_functions.plotting_wrapper_functions import benchmark_wrapper, BenchmarkingResultsFileNames
from tests.create_test_spectra import pesticides_test_spectra, ms2deepscore_model


def test_benchmark_wrapper(tmp_path):
    spectra = pesticides_test_spectra()
    benchmarking_results_file_names = BenchmarkingResultsFileNames(tmp_path, "negative_negative")
    benchmark_wrapper(spectra, spectra, benchmarking_results_file_names, ms2deepscore_model())
