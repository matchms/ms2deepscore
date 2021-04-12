# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/matchms/ms2deepscore/compare/0.2.0...HEAD
[0.2.0]: https://github.com/matchms/ms2deepscore/compare/0.1.3...0.2.0
[0.1.3]: https://github.com/matchms/ms2deepscore/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/matchms/ms2deepscore/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/matchms/ms2deepscore/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/matchms/ms2deepscore/releases/tag/0.1.0
