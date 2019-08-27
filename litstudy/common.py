
class DocumentSet:
    """ Set of documents from query """
    pass


class Document:
    """Example of document"""

    def __init__(self, **kwargs):
        """ Initialize document """

        self.doi = kwargs.pop('doi', None)
        """ Digital Object Identifier (DOI) of the document, or `None` if unavailable. """

        self.title = kwargs.pop('title')
        """ Title of document. """

        self.authors = kwargs.pop('authors', None)
        """ Authors of document as list of `Author` objects, or `None` if unavailable. """

        self.keywords = kwargs.pop('keywords', None)
        """ List of author specified keywords, or `None` if unavailable. """

        self.abstract = kwargs.pop('abstract', None)
        """ Abstract of document, or `None` if unavailable. """

        self.references = kwargs.pop('references', None)
        """ """

        self.year = kwargs.pop('year', None)
        """ Year of publication as integer. """

        self.source = kwargs.pop('source', None)
        """ Name of source (for example: 'The International Conference on Knowledge Discovery and Data Mining') or `None` if unavailable. """

        self.citation_count = kwargs.pop('citation_count', None)
        """ The number of received citations, or `None` if unavailable. """

        self._internal = kwargs.pop('internal', None)
        """ Internal object used to extract these properties  """


class Author:
    def __init__(self, **kwargs):
        self.orcid = kwargs.pop('orcid', None)
        """The ORCID id of the author, or `None` if unavailable."""

        self.name = kwargs.pop('name')
        """Name and surname of the author."""

        self.affiliations = kwargs.pop('affiliations')
        """Affiliations of the author as list of `Affiliation` objects, or `None` if unavailable."""


class Affiliation:
    def __init__(self, **kwargs):
        self.name = kwargs.pop("name")
        """The name of the institution."""

        self.city = kwargs.pop("city", None)
        """The city where the institution is, or `None` if unavailable."""

        self.country = kwargs.pop("country", None)
        """The country where the institution is, or `None` if unavailable."""
