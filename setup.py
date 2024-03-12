#!/usr/bin/env python
import os
from setuptools import find_packages, setup


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
    long_description_content_type='text/markdown',
    author="Florian Huber (as part of the ms2deepscore developer team)",
    author_email="florian.huber@hs-duesseldorf.de",
    url="https://github.com/iomega/ms2deepscore",
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    test_suite="tests",
    python_requires='>=3.8',
    install_requires=[
        "matchms>=0.14.0",
        "numba",
        "numpy>= 1.20.3",
        "pandas",
        "tensorflow<2.14",
        "tqdm",
    ],
    extras_require={"train": ["rdkit"],
                    "dev": ["bump2version",
                            "isort>=4.2.5,<5",
                            "pylint!=2.15.7",
                            "prospector[with_pyroma]",
                            "pytest",
                            "pytest-cov",
                            "sphinx>=3.0.0,!=3.2.0,<4.0.0",
                            "sphinx_rtd_theme",
                            "sphinxcontrib-apidoc",
                            "yapf",],
                    }
)
