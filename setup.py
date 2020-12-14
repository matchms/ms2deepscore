#!/usr/bin/env python
import os

from setuptools import setup
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "ms2deepscore", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="ms2deepscore",
    version=version["__version__"],
    description="Deep learning similarity measure for comparing MS/MS spectra.",
    long_description=readme,
    author="Netherlands eScience Center",
    author_email="f.huber@esciencecenter.nl",
    url="https://github.com/iomega/ms2deepscore",
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    test_suite="tests",
    python_requires='>=3.7',
    install_requires=[
        "matchms",
        "numpy",
        "pandas",
        "tensorflow",
        "tqdm",
    ],
    extras_require={"dev": ["bump2version",
                            "isort>=4.2.5,<5",
                            "prospector[with_pyroma]",
                            "pytest",
                            "pytest-cov",
                            "sphinx>=3.0.0,!=3.2.0,<4.0.0",
                            "sphinx_rtd_theme",
                            "sphinxcontrib-apidoc",
                            "yapf",],
    }
)
