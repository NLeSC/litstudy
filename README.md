# LitStudy

![Logo](https://raw.githubusercontent.com/nlesc/litstudy/master/docs/logo.png#gh-light-mode-only)

[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/NLeSC/litstudy/)
[![DOI](https://zenodo.org/badge/206312286.svg)](https://zenodo.org/badge/latestdoi/206312286)
[![License](https://img.shields.io/github/license/nlesc/litstudy)](https://github.com/NLeSC/litstudy/blob/master/LICENSE)
[![Version](https://img.shields.io/pypi/v/litstudy)](https://pypi.org/project/litstudy/)
[![Build and Test](https://github.com/NLeSC/litstudy/actions/workflows/python-app.yml/badge.svg)](https://github.com/NLeSC/litstudy/actions/)

**LitStudy** is a Python package for analyzing scientific literature right from the comfort of a Jupyter Notebook. It lets you gather publications and explore their metadata through visualizations, network analysis, and natural-language processing.

The package offers five main features:

* **Extract** metadata from scientific documents sourced from various locations. A uniform interface allows combining different data sources.
* **Filter**, **select**, **deduplicate**, and **annotate** document collections.
* Compute and plot general **statistics** for document sets (authors, venues, publication years, and more).
* Generate and plot various **bibliographic networks** as interactive visualizations.
* Discover topics using **natural-language processing** (NLP) to automatically identify popular themes.

## Frequently Asked Questions

If you have any questions or run into an error, see the [_Frequently Asked Questions_](https://nlesc.github.io/litstudy/faq.html) section of the [documentation](https://nlesc.github.io/litstudy/).
If your question or error is not on the list, please check the [GitHub issue tracker](https://github.com/NLeSC/litstudy/issues) for a similar issue or
create a [new issue](https://github.com/NLeSC/litstudy/issues/new).

## Supported Source

LitStudy supports the following data sources. The table below lists which metadata is fully (✓) or partially (*) provided by each source.

| Name            | Title | Authors | Venue | Abstract | Citations      | References |
|-----------------|-------|---------|-------|----------|----------------|------------|
| [Scopus]          | ✓     | ✓       | ✓     | ✓        | ✓              | ✓
| [SemanticScholar] | ✓     | ✓       | ✓     | ✓        | * <sup>(count only)</sup> | ✓
| [CrossRef]        | ✓     | ✓       | ✓     | ✓        | * <sup>(count only)</sup> | ✓
| [DBLP]            | ✓     | ✓       | ✓     |          |                |
| [arXiv]           | ✓     | ✓       |       | ✓        |                |
| [IEEE Xplore]     | ✓     | ✓       | ✓     | ✓        | * <sup>(count only)</sup> |
| [Springer Link]   | ✓     | ✓       | ✓     | ✓        | * <sup>(count only)</sup> |
| CSV file        | ✓     | ✓       | ✓     | ✓        |                |
| bibtex file     | ✓     | ✓       | ✓     | ✓        |                |
| RIS file        | ✓     | ✓       | ✓     | ✓        |                |

[Scopus]: http://scopus.com/
[SemanticScholar]: https://www.semanticscholar.org/
[CrossRef]: https://www.crossref.org/
[DBLP]: https://dblp.org/
[arXiv]: https://arxiv.org/
[IEEE Xplore]: https://ieeexplore.ieee.org/
[Springer Link]: https://link.springer.com/

## Example

An example notebook is available in `notebooks/example.ipynb` and [here](https://nlesc.github.io/litstudy/example.html).

[![Example notebook](https://raw.githubusercontent.com/NLeSC/litstudy/master/docs/images/notebook.png)](https://github.com/NLeSC/litstudy/blob/master/notebooks/example.ipynb)

## Installation Guide

LitStudy is available on PyPI!
Full installation guide is available [here](https://nlesc.github.io/litstudy/installation.html).

```bash
pip install litstudy
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/NLeSC/litstudy
```

## Documentation

Documentation is available [here](https://nlesc.github.io/litstudy/).

## Requirements

The package has been tested for Python 3.7. Required packages are available in `requirements.txt`.

`litstudy` supports several data sources.
Some of these sources (such as semantic Scholar, CrossRef, and arXiv) are openly available.
However to access the Scopus API, you (or your institute) requires a Scopus subscription and you need to request an Elsevier Developer API key (see [Elsevier Developers](https://dev.elsevier.com/index.jsp)).
For more information, see the [guide](https://pybliometrics.readthedocs.io/en/stable/access.html) by `pybliometrics`.

## License

Apache 2.0. See [LICENSE](https://github.com/NLeSC/litstudy/blob/master/LICENSE).

## Change log

See [CHANGELOG.md](https://github.com/NLeSC/litstudy/blob/master/CHANGELOG.md).

## Contributing

See [CONTRIBUTING.md](https://github.com/NLeSC/litstudy/blob/master/CONTRIBUTING.md).

## Citation

If you use LitStudy in your work, please cite the following publication:

> S. Heldens, A. Sclocco, H. Dreuning, B. van Werkhoven, P. Hijma, J. Maassen & R.V. van Nieuwpoort (2022), "litstudy: A Python package for literature reviews", SoftwareX 20

As BibTeX:

```Latex
@article{litstudy,
    title = {litstudy: A Python package for literature reviews},
    journal = {SoftwareX},
    volume = {20},
    pages = {101207},
    year = {2022},
    issn = {2352-7110},
    doi = {https://doi.org/10.1016/j.softx.2022.101207},
    url = {https://www.sciencedirect.com/science/article/pii/S235271102200125X},
    author = {S. Heldens and A. Sclocco and H. Dreuning and B. {van Werkhoven} and P. Hijma and J. Maassen and R. V. {van Nieuwpoort}},
}
```

## Related work

Don't forget to check out these other amazing software packages!

* [ScientoPy](https://www.scientopy.com/): Open-source Python based scientometric analysis tool.
* [pybliometrics](https://github.com/pybliometrics-dev/pybliometrics): API-Wrapper to access Scopus.
* [ASReview](https://asreview.nl/): Active learning for systematic reviews.
* [metaknowledge](https://github.com/UWNETLAB/metaknowledge): Python library for doing bibliometric and network analysis in science.
* [tethne](https://github.com/diging/tethne): Python module for bibliographic network analysis.
* [VOSviewer](https://www.vosviewer.com/): Software tool for constructing and visualizing bibliometric networks.
