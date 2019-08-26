

class Document:
  """Example of document"""

  def __init__(self, **kwargs):
    """ Initialize document """

    self.doi = kwargs.pop('doi', None)
    """ Digital Object Identifier (DOI) of the document, or `None` if unavailable. """

    self.title = kwargs.pop('title', None)
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
    """ The number of received citations, or None if unavailable. """

    self._internal = kwargs.pop('internal', None)
    """ Internal object used to extract these properties  """

class Author:
  def __init__(self, **kwargs):
    self.orcid = kwargs.pop('orcid')
    self.name = kwargs.pop('name')
    self.affiliations = kwargs.pop('affiliations')

class Affiliation:
  #name
  #city
  #country
  pass
  
