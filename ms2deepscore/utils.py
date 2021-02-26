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
