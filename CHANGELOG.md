# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.5.2] - 2025-05-26
### Changed
- Switch to Numpy>=2.0 (which means Python >=3.10)

## [2.5.1] - 2025-02-12
### Changed
* Alternative remove_diagonal implementation by @florian-huber in https://github.com/matchms/ms2deepscore/pull/264
* Restrict torch until new model load/save is implemented by @florian-huber in https://github.com/matchms/ms2deepscore/pull/265

## [2.5.0] - 2024-12-20
### Changed
* Remove outdated notebooks and update notebooks for ms2deepscore 2 by @niekdejonge in https://github.com/matchms/ms2deepscore/pull/238
* Add notebook for optimizing sampling algorithm by @niekdejonge in https://github.com/matchms/ms2deepscore/pull/256
* Added saving history as json file (instead of only plotting) by @niekdejonge in https://github.com/matchms/ms2deepscore/pull/257
* Add notebook hyperparameter optimization by @niekdejonge in https://github.com/matchms/ms2deepscore/pull/258
* Add notebooks benchmarking by @niekdejonge in https://github.com/matchms/ms2deepscore/pull/259
* Add notebooks benchmarking by @niekdejonge in https://github.com/matchms/ms2deepscore/pull/261
* Added notebook for comparison with modified cosine scores by @niekdejonge in https://github.com/matchms/ms2deepscore/pull/262
* Add notebooks analyzing the case studies by @niekdejonge in https://github.com/matchms/ms2deepscore/pull/255

## [2.4.0] - 2024-11-08

## Changed
- Adapted sampling strategy to avoid biases even further [#253](https://github.com/matchms/ms2deepscore/pull/253)

## [2.3.0] - 2024-10-30

### Added
- New plotting functions for benchmarking [#244](https://github.com/matchms/ms2deepscore/pull/244)

### Changed
- Integrated new plotting functions in automated training pipeline [#244](https://github.com/matchms/ms2deepscore/pull/244)
- Removed automatic storing of benchmarking scores [#244](https://github.com/matchms/ms2deepscore/pull/244)
- Integrated loss calculation for validation loss and plots [#244](https://github.com/matchms/ms2deepscore/pull/244)
- Validation loss uses all spectrum pairs instead of only 1 spectrum per inchikey [#244](https://github.com/matchms/ms2deepscore/pull/244)

### Fixed
- Fingerprint type and number of bits specified in the settings is now correctly used in training and validation (before was set to default values in some instances) [#244](https://github.com/matchms/ms2deepscore/pull/244) and [#251](https://github.com/matchms/ms2deepscore/pull/251)

### Removed
- Removed version warning

## [2.2.0] - 2024-10-17

### Changed
- Switch linter (and linting style) from pylint + prospector to ruff [#240](https://github.com/matchms/ms2deepscore/pull/240)
- Clearer documentation and naming to run training from existing splits [#239](https://github.com/matchms/ms2deepscore/pull/239)

### Fixed
- Fixed one memory leak when running the `parameter_serch` function (there might be more though) [#243](https://github.com/matchms/ms2deepscore/pull/243)

## [2.1.0] - 2024-10-07

### Fixed
- A bug of spectrum pair sampling during training was fixed. Due to this bug for each spectrum only one unique spectrum was sampled, even if multiple spectra were available. The bug was introduced with MS2Deepscore 2.0

### Changed
- The inchikey pair selection and data generator has been refactored. The new data generator results in a more balanced inchikey distribution. For details see [#232](https://github.com/matchms/ms2deepscore/pull/232)
- dense layers are now build with leaky ReLU instead of ReLU [#222](https://github.com/matchms/ms2deepscore/pull/222).
- The zenodo link to the latest model has been updated to a model trained using the new algorithm.

### Added
- Missing code documentation [#222](https://github.com/matchms/ms2deepscore/pull/222).

## [2.0.0] - date...
Large scale expansion, revision, and restructuring of MS2Deepscore.

### Added
- Models are now build using PyTorch.
- Models have build-in GPU support (using pytorch).
- new `EmbeddingEvaluatorModel` (Inception Time CNN)
- new `LinearModel` for absolute error estimates
- new `MS2DeepScoreEvaluated` matchms-style score --> gives "score" and "predicted_absolute_error"
- Additional smart binning layer that can handle input of much higher peak resolution (not used as a default!)
- New validation concept --> all-vs-all scores for the validation spectra are computed, but loss is then computed per score bin. This gives better and more significant statistics of the model performance
- New loss functions "Risk Aware MAE" and "Risk Aware MSE" which function similar to MAE or MSE but try to counteract the tendency of a model to predict towards 0.5.
- Losses can now be weighted with a weighting_factor.


### Changed
- No longer supports Tensorflow/Keras
- The concept of Spectrum binning has changed and is now implemented differently (i.e. no more "missing peaks" as before)
- Monte-Carlo Dropout does not return a score (mean or median) together with percentile-based upper and lower bound (instead of STD or IQR before).

## [Unreleased]

## [1.0.0] - 2024-03-12

Last version using Tensorflow. Next versions will be using PyTorch.

### Added
- Added split_positive_and_negative_mode.py [#148](https://github.com/matchms/ms2deepscore/pull/148)
- Added SettingMS2Deepscore [#151](https://github.com/matchms/ms2deepscore/pull/151)
- Clearer Warnings when too little input spectra are used in data generator. [#155](https://github.com/matchms/ms2deepscore/issues/155)

### Changed
- Change the max oversampling rate to max_pairs_per_bin [#148](https://github.com/matchms/ms2deepscore/pull/148)
- Made spectrum pair selection a lot simpler and fixed mistake [#148](https://github.com/matchms/ms2deepscore/pull/148)
- Use DataGeneratorCherrypicked instead of DataGeneratorAllInchikeys in pipelines [#148](https://github.com/matchms/ms2deepscore/pull/148)
- Removed M1 Chip compatibility which lead to faulty results depending on Tensorflow version [#200](https://github.com/matchms/ms2deepscore/pull/200)

   
## [0.5.0] - 2023-08-18

### Added

- New `DataGeneratorCherrypicked` as alternative to former data generators [#145](https://github.com/matchms/ms2deepscore/pull/145). This will work better for large datasets and also tried to counteract biases in the chemical similarity scores.
- Models can now be trained on selected metadata entries in addition to the spectrum peaks [#128](https://github.com/matchms/ms2deepscore/pull/128).
- New `MetadataFeatureGenerator` class to handle additional metadata more robustly [#128](https://github.com/matchms/ms2deepscore/pull/128)
- Workflow scripts for training a new MS2DeepScore model [#124](https://github.com/matchms/ms2deepscore/pull/124). The ease of training MS2Deepscore models is improved, including standard settings and splitting validation and training data.

### Changed

- In SiameseModel, the attributes are not passed as an argument but instead used by the class.
- Improved plotting functionality. Some additional plotting options were added and plots previously created in notebooks are now functions.
- Linting (code and imports) [#145](https://github.com/matchms/ms2deepscore/pull/145).


## [0.4.0] - 2023-04-25

### Added

- Functions to cover the full pipeline of training a new model [#129](https://github.com/matchms/ms2deepscore/pull/129)

### Fixed

- Tensorflow issues when saving/loading models [#123](https://github.com/matchms/ms2deepscore/issues/123)

### Changed

- Random seed is now optional when `fixed_set=True` for the data generator [#134](https://github.com/matchms/ms2deepscore/pull/134)
- `load_model()` functions now auto-detects if a model is multi_inputs or not
- Python version support was changed to 3.8, 3.9, 3.10 (other versions should still work but are not systematically tested)

## [0.3.1] - 2023-01-06

### Changed

- Minor changes to make tests work with new matchms (>=0.18.0). Older versions should work as well though. [#120](https://github.com/matchms/ms2deepscore/pull/120)

## [0.3.0] - 2022-11-29

## Added

- Allow adding metadata to the network inputs, e.g. precursor-m/z using the `additional_inputs` parameter [#115](https://github.com/matchms/ms2deepscore/pull/115)

## Fixed

- Update test to work with Tensorflow 2.11 [#114](https://github.com/matchms/ms2deepscore/pull/114)

## [0.2.3] - 2022-03-02

## Fixed

- Fixes issue [#97](https://github.com/matchms/ms2deepscore/pull/97) by raising a ValueError when duplicate InChiKey14 are specified by the user in the reference_scores_df DataFrame.

## Changed

- Minor linting [#93](https://github.com/matchms/ms2deepscore/pull/93)

## Fixed

- Handled numby dependency issues [#94](https://github.com/matchms/ms2deepscore/issues/94) and [#95](https://github.com/matchms/ms2deepscore/issues/95)

## [0.2.2] - 2021-08-19

## Fixed

- now compatible with new Tensorflow 2.6, also checked by additional CI runs for Tensorflow 2.4, 2.5 and 2.6 [#92](https://github.com/matchms/ms2deepscore/pull/92)

## [0.2.1] - 2021-07-20

## Changed

- Speed improvement of spectrum binning step [#90](https://github.com/matchms/ms2deepscore/pull/90)

## [0.2.0] - 2021-04-01

## Added

- `MS2DeepScoreMonteCarlo` Monte-Carlo dropout based ensembling do obtain mean/median score and STD [#65](https://github.com/matchms/ms2deepscore/pull/65)
- choice between `median` (default) and `mean` ensemble score which come with `IQR` and `STD` as uncertainty measures [#86](https://github.com/matchms/ms2deepscore/pull/86)
- `dropout_in_first_layer` option for SiameseModel (default is False) [#86](https://github.com/matchms/ms2deepscore/pull/86)
- `use_fixed_set` option for data generators to create deterministic training/testing data with fixed random seed [#73](https://github.com/matchms/ms2deepscore/issues/73)

## Changed

- small update of `create_histograms_plot` to make the plot prettier/better to read [#85](https://github.com/matchms/ms2deepscore/pull/85)

## Fixed

- solved minor unclarity with the pair selection for non-available reference scores [#79](https://github.com/matchms/ms2deepscore/pull/79)
- solved minor unclarity with the addition of noise peaks during data augmentation [#78](https://github.com/matchms/ms2deepscore/pull/78)

## [0.1.3] - 2021-03-09

## Changed

- Allow users to define L1 and L2 regularization of `SiameseModel` [#67](https://github.com/matchms/ms2deepscore/issues/67)
- Allow users to define number and size of `SiameseModel` [#64](https://github.com/matchms/ms2deepscore/pull/64)

## [0.1.2] - 2021-03-05

## Added

- `create_confusion_matrix_plot` in `plotting` [#58](https://github.com/matchms/ms2deepscore/pull/58)

## [0.1.1] - 2021-02-09

## Added

- noise peak addition during training via data generators [#55](https://github.com/matchms/ms2deepscore/pull/55)
- L1 and L2 regularization for first dense layer [#55](https://github.com/matchms/ms2deepscore/pull/55)

## Changed

- move vector calculation to separate calculate_vectors method [#52](https://github.com/matchms/ms2deepscore/pull/52)

## [0.1.0] - 2021-02-08

### Added

- This is the initial version of MS2DeepScore

[Unreleased]: https://github.com/matchms/ms2deepscore/compare/2.5.2...HEAD
[2.5.2]: https://github.com/matchms/ms2deepscore/compare/2.5.1...2.5.2
[2.5.1]: https://github.com/matchms/ms2deepscore/compare/2.5.0...2.5.1
[2.5.0]: https://github.com/matchms/ms2deepscore/compare/2.4.0...2.5.0
[2.4.0]: https://github.com/matchms/ms2deepscore/compare/2.3.0...2.4.0
[2.3.0]: https://github.com/matchms/ms2deepscore/compare/2.2.0...2.3.0
[2.2.0]: https://github.com/matchms/ms2deepscore/compare/2.1.0...2.2.0
[2.1.0]: https://github.com/matchms/ms2deepscore/compare/2.0.0...2.1.0
[2.0.0]: https://github.com/matchms/ms2deepscore/compare/1.0.0...2.0.0
[1.0.0]: https://github.com/matchms/ms2deepscore/compare/0.5.0...1.0.0
[0.5.0]: https://github.com/matchms/ms2deepscore/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/matchms/ms2deepscore/compare/0.3.1...0.4.0
[0.3.1]: https://github.com/matchms/ms2deepscore/compare/0.3.1...0.3.1
[0.3.0]: https://github.com/matchms/ms2deepscore/compare/0.2.3...0.3.0
[0.2.3]: https://github.com/matchms/ms2deepscore/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/matchms/ms2deepscore/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/matchms/ms2deepscore/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/matchms/ms2deepscore/compare/0.1.3...0.2.0
[0.1.3]: https://github.com/matchms/ms2deepscore/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/matchms/ms2deepscore/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/matchms/ms2deepscore/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/matchms/ms2deepscore/releases/tag/0.1.0
