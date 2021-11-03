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

## Reference
If you use MS2DeepScore for your research, please cite the following:

**"MS2DeepScore - a novel deep learning similarity measure to compare tandem mass spectra"**
Florian Huber, Sven van der Burg, Justin J.J. van der Hooft, Lars Ridder, 13, Article number: 84 (2021), Journal of Cheminformatics, doi: https://doi.org/10.1186/s13321-021-00558-4


## Setup
### Requirements

Python 3.7 or higher

### Installation
Simply install using pip: `pip install ms2deepscore`

### Prepare environment
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

## Getting started: How to prepare data, train a model, and compute similarities.
See [notebooks/MS2DeepScore_tutorial.ipynb](https://github.com/matchms/ms2deepscore/blob/main/notebooks/MS2DeepScore_tutorial.ipynb) 
for a more extensive fully-working example on test data.
If you are not familiar with `matchms` yet, then we also recommand our [tutorial on how to get started using matchms](https://blog.esciencecenter.nl/build-your-own-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-i-d96c718c68ee).

There are two different ways to use MS2DeepScore to compute spectral similarities. You can train a new model on a dataset of your choice. That, however, should preferentially contain a substantial amount of spectra to learn relevant features, say > 10,000 spectra of sufficiently diverse types.
The second way is much simpler: Use a model that was pretrained on a large dataset. 

## 1) Use a pretrained model to compute spectral similarities
We provide a model which was trained on > 100,000 MS/MS spectra from [GNPS](https://gnps.ucsd.edu/), which can simply be downloaded [from zenodo here](https://zenodo.org/record/4699356).
To then compute the similarities between spectra of your choice you can run something like:
```python
from matchms import calculate_scores()
from matchms.importing import load_from_msp
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

# Import data
references = load_from_msp("my_reference_spectra.msp")
queries = load_from_msp("my_query_spectra.msp")

# Load pretrained model
model = load_model("MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5")

similarity_measure = MS2DeepScore(model)
# Calculate scores and get matchms.Scores object
scores = calculate_scores(references, queries, similarity_measure)
```

If you want to calculate all-vs-all spectral similarities, e.g. to build a network, than you can run:
```python
scores = calculate_scores(references, references, similarity_measure, is_symmetric=True)
```

To use Monte-Carlo Dropout to also get a uncertainty measure with each score, run the following:
```python
from matchms import calculate_scores()
from matchms.importing import load_from_msp
from ms2deepscore import MS2DeepScoreMonteCarlo
from ms2deepscore.models import load_model

# Import data
references = load_from_msp("my_reference_spectra.msp")
queries = load_from_msp("my_query_spectra.msp")

# Load pretrained model
model = load_model("MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5")

similarity_measure = MS2DeepScoreMonteCarlo(model, n_ensembles=10)
# Calculate scores and get matchms.Scores object
scores = calculate_scores(references, queries, similarity_measure)
```
In that scenario, `scores["score"]` contains the similarity scores (median of the ensemble of 10x10 scores) and `scores["uncertainty"]` give an uncertainty estimate (interquartile range of ensemble of 10x10 scores.

## 2) Train an own MS2DeepScore model
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
We welcome contributions to the development of ms2deepscore! Have a look at the [contribution guidelines](https://github.com/matchms/ms2deepscore/blob/main/CONTRIBUTING.md).
