import logging
from .__version__ import __version__
from .SpectrumBinner import SpectrumBinner


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__all__ = [
    "__version__",
    "SpectrumBinner",
]
