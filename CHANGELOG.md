# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed
### Removed
### Fixed
- Fix bug in the semantic scholar backend that did not fetch papers correctly
- Fix bug in `fetch_crossref` where document title was not extracted correctly

## [1.0.2] - 2022-05-25

### Changed
- Remove dependency on fa2. The version of fa2 on pip is broken under Python 3.9+.

### Fixed
- `litstudy` now works under Python 3.9+.


## [1.0.1] - 2022-05-16

### Added
- Support for the arXiv API (Thanks ksilo!)

### Changed
- Made project compatible with Python 3.6


## [1.0.0] - 2022-02-17

### Changed

- Complete rewrite of litstudy project for 1.0 release.
- Rename from "automated-literature-analysis" to "litstudy".

## [0.0.1] - 2019-09-04

### Added
Initial release of litstudy

