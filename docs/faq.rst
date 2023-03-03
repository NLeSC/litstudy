Frequently Asked Questions
==========================

This pages lists answers to several common issues that can occur when working with LitStudy.
If your question is not in this list, please create an issue on the `GitHub issue tracker <https://github.com/NLeSC/litstudy/issues>`_.


How to use Scopus?
---------------------
To use the Scopus API, you will need two things:

 * An Elsevier API key obtainable through the `Elsevier Developer Portal <https://dev.elsevier.com/>`_. You or (your institute) must require a Scopus subscription.
 * Be connected to the network of your University or Research Institute for which you obtained the API key.

LitStudy will ask for the API key on the first time that it launches.



I'm having trouble connecting to Scopus!
----------------------------------------

LitStudy internally uses the Python package `pybliometrics <https://pybliometrics.readthedocs.io/en/stable/configuration.html>`_ to communicate with the Scopus API.
See the page on `pybliometrics configuration <https://pybliometrics.readthedocs.io/en/stable/configuration.html>`_ for more information.

Alternatively, you can use one of the free alternatives to Scopus (see :doc:`api/sources`) such as, for example, SemanticScholar (``litstudy.search_semanticscholar``) or CrossRef (``litstudy.search_crossref``).


Scopus400Error: ``Exceeds the maximum number allowed for the service level``
----------------------------------------------------------------------------
You Scopus query returns too many results. Please limit your query, for example, by restricting the publication year using ``... AND PUBYEAR > 2020``.
It could also be the case that your Scopus API key is invalid, in which case see `How to use Scopus?`


Scopus401Error: ``The requestor is not authorized to access the requested view or fields of the resource``
----------------------------------------------------------------------------------------------------------
It is likely that your Scopus API key is invalid, in which case see `How to use Scopus?`


My question is not in this list?
--------------------------------
If your question is not in this list, please create a new issue on `GitHub <https://github.com/NLeSC/litstudy/issues/new>`_.
