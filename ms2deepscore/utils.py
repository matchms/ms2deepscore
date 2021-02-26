import re


def create_peak_dict(peak_list):
    """ Create dictionary of merged peaks (keep max-intensity peak per bin).
    """
    peaks = {}
    for (ID, weight) in peak_list:
        if ID in peaks:
            peaks[ID] = max(weight, peaks[ID])
        else:
            peaks[ID] = weight
    return peaks


def is_valid_inchikey14(inchikey14: str) -> bool:
 """Return True if string has format of inchikey14.

 Parameters
 ----------
 inchikey14
     Input string to test if it format of an inchikey14.
 """
 if inchikey14 is None:
     return False

 regexp = r"[A-Z]{14}"
 if re.fullmatch(regexp, inchikey14):
     return True
 return False
