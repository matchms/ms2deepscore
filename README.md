![GitHub](https://img.shields.io/github/license/matchms/ms2deepscore)
[![PyPI](https://img.shields.io/pypi/v/ms2deepscore)](https://pypi.org/project/ms2deepscore/)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/matchms/ms2deepscore/CI%20Build)
[![SonarCloud Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=matchms_ms2deepscore&metric=alert_status)](https://sonarcloud.io/dashboard?id=matchms_ms2deepscore)
[![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=matchms_ms2deepscore&metric=coverage)](https://sonarcloud.io/component_measures?id=matchms_ms2deepscore&metric=Coverage&view=list)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

# ms2deepscore
Deep learning similarity measure for comparing MS/MS spectra with respect to their chemical similarity

## Requirements

Python 3.7 or higher

## Prepare environmnent
We recommend to create an Anaconda environment with

```
conda create --name ms2deepscore python=3.8
conda activate ms2deepscore
pip install ms2deepscore
```
Alternatively, simply install in the environment of your choice by `pip install ms2deepscore`.


Or, to have the full matchms functionality:
```
conda create --name ms2deepscore python=3.8
conda activate ms2deepscore
conda install --channel nlesc --channel bioconda --channel conda-forge matchms
pip install ms2deepscore
```
