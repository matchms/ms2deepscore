import pytest
import ms2deepscore.utils as utils


def test_save_and_load_pickled_file(tmpdir):
    temp_file = tmpdir.join("temp.pkl")
    data = {"test_key": "test_value"}
    
    # Saving and loading without any issues
    utils.save_pickled_file(data, temp_file.strpath)
    loaded_data = utils.load_pickled_file(temp_file.strpath)
    assert data == loaded_data

    # Asserting that save_pickled_file raises an exception if file already exists
    with pytest.raises(FileExistsError, match="File already exists"):
        utils.save_pickled_file(data, temp_file.strpath)


def test_return_non_existing_file_name(tmpdir):
    base_filename = tmpdir.join("test_file.txt")
    first_duplicate = tmpdir.join("test_file(1).txt")

    # If the file doesn't exist, it should return the same filename
    assert utils.return_non_existing_file_name(base_filename) == base_filename

    # Create the base file and test again
    open(base_filename, "w").close()
    assert utils.return_non_existing_file_name(base_filename) == first_duplicate

    # Create the first duplicate and test again
    open(first_duplicate, "w").close()
    assert utils.return_non_existing_file_name(base_filename) == tmpdir.join("test_file(2).txt")

@pytest.mark.parametrize("bins,correct", [
    ([(-0.01, 1)], True),
    ([(0.8, 0.9), (0.7, 0.8), (0.9, 1.0), (0.6, 0.7), (0.5, 0.6),
      (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (-0.01, 0.1)], True),
    ([(-0.01, 0.6), (0.7, 1.0)], False),  # Test a gap in bins is detected
    ([(0.0, 0.6), (0.7, 1.0)], False),  # Test that the lowest values is below 0.
    ([(-0.3, -0.1), (-0.1, 1.0)], False),  # Test that no bin is entirely below 0.
    ([(0.0, 0.6), (0.6, 0.6), (0.6, 1.0)], False),  # Test no repeating bin borders
    ([(0.0, 0.6), (0.7, 0.6), (0.7, 1.0)], False),  # Test correct order of bin borders
    ([(0.0, 0.5, 1.), (0.5, 0.7, 1.), (0.7, 1.0)], False),  # Test all bins have two elements
])
def test_validate_bin_order(bins, correct):
    if correct:
        utils.validate_bin_order(bins)
    else:
        with pytest.raises(ValueError):
            utils.validate_bin_order(bins)
