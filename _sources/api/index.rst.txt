API reference
====================================

This page provides the API documentation of litstudy.
All public functions are re-exported under the global `litstudy` namespace for convenience.
However, the code is structured hierachical meaning the documentation shows the hierarchical names.
For example:

.. code-block:: python

    docs = litstudy.sources.scopus.search_scopus("example")
    litstudy.plot.plot_author_histogram(docs)

    # Is equivalent to

    docs = litstudy.search_scopus("example")
    litstudy.plot_author_histogram(docs)


The package is divided into 6 modules:

* Core data types such as `Document` and `DocumentSet`.
* Functions to retrieve or load scientific citations.
* Compute general statistics.
* Generate bibliographic networks.
* Automatic topic detection using natural language processing (NLP).
* Plot results. These functions are mostly useful inside a notebook.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   types
   sources
   stats
   network
   nlp
   plot
