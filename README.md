![GitHub](https://img.shields.io/github/license/matchms/ms2deepscore)
[![PyPI](https://img.shields.io/pypi/v/ms2deepscore?color=teal)](https://pypi.org/project/ms2deepscore/)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/matchms/ms2deepscore/CI_build.yml?branch=main)
[![SonarCloud Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=matchms_ms2deepscore&metric=alert_status)](https://sonarcloud.io/dashboard?id=matchms_ms2deepscore)
[![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=matchms_ms2deepscore&metric=coverage)](https://sonarcloud.io/component_measures?id=matchms_ms2deepscore&metric=Coverage&view=list)  
[![DOI](https://zenodo.org/badge/310047938.svg)](https://zenodo.org/badge/latestdoi/310047938)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

<img src="https://github.com/matchms/ms2deepscore/blob/main/materials/ms2deepscore_logo.png" width="400">

# ms2deepscore
`ms2deepscore` provides a Siamese neural network that is trained to predict molecular structural similarities (Tanimoto scores) 
from pairs of mass spectrometry spectra. 

The library provides intuitive classes to prepare data, train a Siamese model,
and compute similarities between pairs of spectra.

In addition to the prediction of a structural similarity, 
MS2DeepScore can also make use of Monte-Carlo dropout to assess the model's uncertainty.

## Reference
If you use MS2DeepScore for your research, please cite the following:

**"MS2DeepScore - a novel deep learning similarity measure to compare tandem mass spectra"**
Florian Huber, Sven van der Burg, Justin J.J. van der Hooft, Lars Ridder, 13, Article number: 84 (2021), Journal of Cheminformatics, doi: https://doi.org/10.1186/s13321-021-00558-4

If you use MS2Deepscore 2.0 or higher please also cite:
**Reliable cross-ion mode chemical similarity prediction between MS2 spectra**
Niek de Jonge, David Joas, Lem-Joe Truong, Justin J.J. van der Hooft, Florian Huber
bioRxiv 2024.03.25.586580; doi: https://doi.org/10.1101/2024.03.25.586580


## Setup
### Requirements

Python 3.10, 3.11, 3.12 (higher will likely work but is not tested systematically).

### Installation
Installation is expected to take 10-20 minutes.

### Prepare environment
We recommend creating an Anaconda environment with

```
conda create --name ms2deepscore python=3.9
conda activate ms2deepscore
pip install ms2deepscore
```

Or, via conda:
```
conda create --name ms2deepscore python=3.9
conda activate ms2deepscore
conda install --channel bioconda --channel conda-forge matchms
pip install ms2deepscore
```

Alternatively, simply install in the environment of your choice by `pip install ms2deepscore`

## Getting started: How to prepare data, train a model, and compute similarities.
We recommend to run the complete tutorial in [notebooks/MS2DeepScore_tutorial.ipynb](https://github.com/matchms/ms2deepscore/blob/main/notebooks/tutorials/ms2deepscore_tutorial.ipynb) 
for a more extensive fully-working example on test data. The expected run time on a laptop is less than 5 minutes, including automatic model and dummy data download. 
Alternatively there are some example scripts below.
If you are not familiar with `matchms` yet, then we also recommand our [tutorial on how to get started using matchms](https://blog.esciencecenter.nl/build-your-own-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-i-d96c718c68ee).

## 1) Compute spectral similarities
We provide a model which was trained on > 500,000 MS/MS combined spectra from [GNPS](https://gnps.ucsd.edu/), [Mona](https://mona.fiehnlab.ucdavis.edu/), MassBank and MSnLib. 
This model can be downloaded from [from zenodo here](https://zenodo.org/records/13897744). Only the ms2deepscore_model.pt is needed.
The model works for spectra in both positive and negative ionization modes and even predictions across ionization modes can be made by this model. 

To compute the similarities between spectra of your choice you can run the code below.
There is a small example dataset available in the folder "./tests/resources/pesticides_processed.mgf". 
Alternatively you can of course use your own spectra, most common formats are supported, e.g. msp, mzml, mgf, mzxml, json, usi.
```python
from ms2deepscore.models import load_model
from matchms.Pipeline import Pipeline, create_workflow
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from ms2deepscore import MS2DeepScore

model_file_name = "ms2deepscore_model.pt"
spectrum_file_name = "pesticides.mgf"

# load in the ms2deepscore model
model = load_model(model_file_name)

pipeline = Pipeline(create_workflow(query_filters=DEFAULT_FILTERS,
                                    score_computations=[[MS2DeepScore, {"model": model}]]))
report = pipeline.run(spectrum_file_name)
similarity_matrix = pipeline.scores.to_array()
```
The resulting similarity matrix, is a numpy array containing all the MS2DeepScore predicitons between all spectra.


## 2 Create embeddings

To calculate chemical similarity scores MS2DeepScore first calculates an embedding (vector) representing each spectrum. 
This intermediate product can also be used to visualize spectra in "chemical space" by using a dimensionality reduction technique, like UMAP.

```python
cleaned_spectra = pipeline.spectra_queries

ms2ds_model = MS2DeepScore(model)
ms2ds_embeddings = ms2ds_model.get_embedding_array(cleaned_spectra)
```
The [tutorial](https://github.com/matchms/ms2deepscore/blob/main/notebooks/MS2DeepScore_tutorial.ipynb) shows how to use these embeddings to create an interactive UMAP with overlaying smiles.
<img src="https://github.com/matchms/ms2deepscore/blob/main/materials/umap_example.png" width="400"/>

## 3) Train your own MS2DeepScore model
Training your own model is only recommended if you have some familiarity with machine learning. 
You can train a new model on a dataset of your choice. That, however, should preferentially contain a substantial amount of spectra to learn relevant features, say > 100,000 spectra of sufficiently diverse types.
Alternatively you can add your in house spectra to an already available public library, for instance the [data](https://zenodo.org/records/13934470) used for training the default MS2DeepScore model. 
To train your own model you can run the code below.
Please first ensure cleaning your spectra. We recommend using the cleaning pipeline in [matchms](https://github.com/matchms/matchms).

```python
from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.wrapper_functions.training_wrapper_functions import train_ms2deepscore_wrapper

spectrum_file = "./combined_libraries.mgf"
# The settins below use default training settings and use precursor mz and ionmode as additional metadata input. 
# Have a look in the SettingsMS2Deepscore class to check other hyperparameters.
settings = SettingsMS2Deepscore(
    additional_metadata=[("CategoricalToBinary", {"metadata_field": "ionmode",
                                                  "entries_becoming_one": "positive",
                                                  "entries_becoming_zero": "negative"}),
                         ("StandardScaler", {"metadata_field": "precursor_mz", 
                                             "mean": 0, "standard_deviation": 1000})],)

train_ms2deepscore_wrapper(spectrum_file, settings, validation_split_fraction=20)
```
## Contributing
We welcome contributions to the development of ms2deepscore! Have a look at the [contribution guidelines](https://github.com/matchms/ms2deepscore/blob/main/CONTRIBUTING.md).
