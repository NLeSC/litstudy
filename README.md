# litstudy
[![DOI](https://zenodo.org/badge/206312286.svg)](https://zenodo.org/badge/latestdoi/206312286)
[![License](https://img.shields.io/github/license/nlesc/litstudy)](https://github.com/NLeSC/litstudy/blob/master/LICENSE)
[![Version](https://img.shields.io/pypi/v/litstudy)]([https://pypi.org/project/litstudy/)
[![Github](https://img.shields.io/github/checks-status/nlesc/litstudy/master)](https://github.com/NLeSC/litstudy)

litstudy is a Python package that allows analysis of scientific literature from the comfort of a Jypyter notebook.
It enables selecting scientific publications and study their metadata using visualizations, network analysis, and natural language processing.

In essense, this package offers five features

* Extract metadata of scientific documents from various sources. The data is unitied by a standard interface, allowing data from different sources to be combined.
* Filter, select, dudplibcate, and annotate collections of documents.
* Compute and plot general statistics of document sets (e.g., statistics on authors, venues, publication years, etc.)
* Generate and plot various bibliographic networks as an interactive visualization.
* Topic discovery based on natural language processing (NLP) allows automatic discovery of popular topics.

## Example
An example notebook is available in `notebooks/example.ipynb` and [here](https://nlesc.github.io/litstudy/example.html).

## Documentation

Document is available [here](https://nlesc.github.io/litstudy/).

## Requirements
The package has been tested for Python 3.6. Required packages are available in `requirements.txt`.

To access the `Scopus` API using `litstudy`, you (or your institute) needs a Scopus subscription and you need to request an Elsevier Developer API key (see [Elsevier Developers](https://dev.elsevier.com/index.jsp).

## Running using virtualenv
Installation using `virtualenv` is can be using the following commands:

Create virtualenv environment named myenv:
```
virtualenv myenv --python=`which python3`
```

Activate virtual environment
```
source ./myenv/bin/activate
```

Install requirement dependencies.
```
pip3 install -r requirements.txt
```

Install new Jupyter kernel.
```
ipython kernel install --user --name=myenv
```

Run Jupyter and select `myenv` as kernel. Remaining instructions can be found within the notebook itself.
```
jupyter notebook --MappingKernelManager.default_kernel_name=myenv
```

## License
Apache 2.0. See [LICENSE](https://github.com/NLeSC/litstudy/blob/master/LICENSE).

## Change log
See [CHANGELOG.md](https://github.com/NLeSC/litstudy/blob/master/CHANGELOG.md).

## Contributing
See [CONTRIBUTING.md](https://github.com/NLeSC/litstudy/blob/master/CONTRIBUTING.md).

## Related work

Don't forget to check out these amazing software packages!

* [ScientoPy](https://www.scientopy.com/): Open-source Python based scientometric analysis tool.
* [pybliometrics](https://github.com/pybliometrics-dev/pybliometrics): API-Wrapper to access Scopus.
* [metaknowledge](https://github.com/UWNETLAB/metaknowledge): Python library for doing bibliometric and network analysis in science.
* [tethne](https://github.com/diging/tethne): Python module for bibliographic network analysis.
* [VOSviewer](https://www.vosviewer.com/): Software tool for constructing and visualizing bibliometric networks.
