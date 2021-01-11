class BinnedSpectrum:
    """Binned spectrum for use with MS2DeepScore."""
    def __init__(self, binned_peaks: dict, metadata: dict):
        """

        Parameters
        ----------
        binned_peaks
            Dictionary of binned peaks (format is {"peak ID":  weight})
        metadata
            Dictionary containing spectrum metadata.
        """
        self.binned_peaks = binned_peaks
        self._metadata = metadata

    def __eq__(self, other):
        return \
            self.binned_peaks == other.binned_peaks and \
            self.metadata == other.metadata

    def get(self, key: str, default=None):
        """Retrieve value from :attr:`metadata` dict. Shorthand for

        .. code-block:: python

            val = self.metadata.get("key", default)

        """
        return self._metadata.get(key, default)

    def set(self, key: str, value):
        """Set value in :attr:`metadata` dict. Shorthand for

        .. code-block:: python

            self.metadata[key] = val

        """
        self._metadata[key] = value
        return self

    @property
    def metadata(self):
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
