# litstudy
[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/NLeSC/litstudy/)
[![DOI](https://zenodo.org/badge/206312286.svg)](https://zenodo.org/badge/latestdoi/206312286)
[![License](https://img.shields.io/github/license/nlesc/litstudy)](https://github.com/NLeSC/litstudy/blob/master/LICENSE)
[![Version](https://img.shields.io/pypi/v/litstudy)](https://pypi.org/project/litstudy/)
[![Build and Test](https://github.com/NLeSC/litstudy/actions/workflows/python-app.yml/badge.svg)](https://github.com/NLeSC/litstudy/actions/)

`litstudy` is a Python package that allows analysis of scientific literature from the comfort of a Jupyter notebook.
It enables selecting scientific publications and study their metadata using visualizations, network analysis, and natural language processing.

In essence, this package offers five features

* Extract metadata of scientific documents from various sources. The data is united by a standard interface, allowing data from different sources to be combined.
* Filter, select, deduplicate, and annotate collections of documents.
* Compute and plot general statistics of document sets (e.g., statistics on authors, venues, publication years, etc.)
* Generate and plot various bibliographic networks as an interactive visualization.
* Topic discovery based on natural language processing (NLP) allows automatic discovery of popular topics.


## Example
An example notebook is available in `notebooks/example.ipynb` and [here](https://nlesc.github.io/litstudy/example.html).

[![Example notebook](https://raw.githubusercontent.com/NLeSC/litstudy/master/docs/images/notebook.png)](https://github.com/NLeSC/litstudy/blob/master/notebooks/example.ipynb)


## Installation Guide
litstudy is available on PyPI!
Full installation guide is available [here](https://nlesc.github.io/litstudy/installation.html).

```bash
pip install litstudy
```

Or install the lastest development version directly from GitHub:

```bash
pip install git+https://github.com/NLeSC/litstudy
```


## Documentation

Documentation is available [here](https://nlesc.github.io/litstudy/).


## Requirements
The package has been tested for Python 3.6. Required packages are available in `requirements.txt`.

To access the `Scopus` API using `litstudy`, you (or your institute) needs a Scopus subscription and you need to request an Elsevier Developer API key (see [Elsevier Developers](https://dev.elsevier.com/index.jsp).


## License
Apache 2.0. See [LICENSE](https://github.com/NLeSC/litstudy/blob/master/LICENSE).


## Change log
See [CHANGELOG.md](https://github.com/NLeSC/litstudy/blob/master/CHANGELOG.md).


## Contributing
See [CONTRIBUTING.md](https://github.com/NLeSC/litstudy/blob/master/CONTRIBUTING.md).


## Related work

Don't forget to check out these other amazing software packages!

* [ScientoPy](https://www.scientopy.com/): Open-source Python based scientometric analysis tool.
* [pybliometrics](https://github.com/pybliometrics-dev/pybliometrics): API-Wrapper to access Scopus.
* [ASReview](https://asreview.nl/): Active learning for systematic reviews.
* [metaknowledge](https://github.com/UWNETLAB/metaknowledge): Python library for doing bibliometric and network analysis in science.
* [tethne](https://github.com/diging/tethne): Python module for bibliographic network analysis.
* [VOSviewer](https://www.vosviewer.com/): Software tool for constructing and visualizing bibliometric networks.
