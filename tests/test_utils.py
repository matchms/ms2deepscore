import pytest
import ms2deepscore.utils as utils


def test_create_peak_dict():
    peak_list = [(1, 5), (2, 6), (1, 7)]
    expected = {1: 7, 2: 6}
    result = utils.create_peak_dict(peak_list)
    assert result == expected


def test_save_and_load_pickled_file(tmpdir):
    temp_file = tmpdir.join("temp.pkl")
    data = {"test_key": "test_value"}
    
    # Saving and loading without any issues
    utils.save_pickled_file(data, temp_file.strpath)
    loaded_data = utils.load_pickled_file(temp_file.strpath)
    assert data == loaded_data

    # Asserting that save_pickled_file raises an exception if file already exists
    with pytest.raises(AssertionError, match="File already exists"):
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
