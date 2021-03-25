![GitHub](https://img.shields.io/github/license/matchms/ms2deepscore)
[![PyPI](https://img.shields.io/pypi/v/ms2deepscore)](https://pypi.org/project/ms2deepscore/)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/matchms/ms2deepscore/CI%20Build)
[![SonarCloud Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=matchms_ms2deepscore&metric=alert_status)](https://sonarcloud.io/dashboard?id=matchms_ms2deepscore)
[![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=matchms_ms2deepscore&metric=coverage)](https://sonarcloud.io/component_measures?id=matchms_ms2deepscore&metric=Coverage&view=list)  
[![DOI](https://zenodo.org/badge/310047938.svg)](https://zenodo.org/badge/latestdoi/310047938)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

# ms2deepscore
ms2deepscore provides a Siamese neural network that is trained to predict molecular structural similarities (Tanimoto scores) 
from pairs of mass spectrometry spectra. 

The library provides an intuitive classes to prepare data, train a siamese model,
and compute similarities between pairs of spectra.

In addition to the prediction of a structural similarity, 
MS2DeepScore can also make use of Monte-Carlo dropout to assess the model uncertainty.


## Setup
### Requirements

Python 3.7 or higher

### Installation
Simply install using pip: `pip install ms2deepscore`

###Prepare environment
We recommend to create an Anaconda environment with

```
conda create --name ms2deepscore python=3.8
conda activate ms2deepscore
pip install ms2deepscore
```
Alternatively, simply install in the environment of your choice by .


Or, to also include the full [matchms](https://github.com/matchms/matchms) functionality:
```
conda create --name ms2deepscore python=3.8
conda activate ms2deepscore
conda install --channel bioconda --channel conda-forge matchms
pip install ms2deepscore
```

## Quick start: How to prepare data, train a model, and compute similarities.
See [notebooks/MS2DeepScore_tutorial.ipynb](https://github.com/matchms/ms2deepscore/blob/main/notebooks/MS2DeepScore_tutorial.ipynb) 
for a more extensive fully-working example on test data.

### Data preperation
Bin spectrums using `ms2deepscore.SpectrumBinner`. 
In this binned form we can feed spectra to the model.
```python
from ms2deepscore import SpectrumBinner
spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
binned_spectrums = spectrum_binner.fit_transform(spectrums)
```
Create a data generator that will generate batches of training examples.
Each training example consists of a pair of binned spectra and the corresponding reference similarity score.
```python
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
dimension = len(spectrum_binner.known_bins)
data_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                           dim=dimension)
```
### Train a model
Initialize and train a SiameseModel. 
It consists of a dense 'base' network that produces an embedding for each of the 2 inputs.
The 'head' model computes the cosine similarity between the embeddings.
```python
from tensorflow import keras
from ms2deepscore.models import SiameseModel
model = SiameseModel(spectrum_binner, base_dims=(200, 200, 200), embedding_dim=200,
                     dropout_rate=0.2)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
model.fit(data_generator,
          validation_data=data_generator,
          epochs=2)
```
### Predict similarity scores
Calculate similariteis for a pair of spectra
```python
from ms2deepscore import MS2DeepScore
similarity_measure = MS2DeepScore(model)
score = similarity_measure.pair(spectrums[0], spectrums[1])
```

## Contributing
We welcome contributions to the development of ms2deepscore! Have a look at the [contribution guidelines](ttps://github.com/matchms/ms2deepscore/blob/main/CONTRIBUTING.md).
