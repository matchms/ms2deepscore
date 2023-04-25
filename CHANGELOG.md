# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/matchms/ms2deepscore/compare/0.4.0...HEAD
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
