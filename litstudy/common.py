
class DocumentSet:
    """ Set of documents from retrieved from search query. """

    def __init__(self, docs):
        self.docs = docs
        """ List of documents as `Document` objects. """

class Document:
  """ Meta data of academic document. """

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

    if kwargs:
        raise KeyError('got an unexpected keyword argument {}'.format(next(iter(kwargs))))

class Author:
    """ Author of `Document` """
    
    def __init__(self, **kwargs):
        self.orcid = kwargs.pop('orcid', None)
        """ ORCID iD of author or `None` if unavailable` """

        self.name = kwargs.pop('name', None)
        """ Name of author as reported by provider or `None` if unavailable. """

        self.affiliations = kwargs.pop('affiliations', None)
        """ Affiliations of authors as list of `Affiliation`, or `None` if unavailable. """

        if kwargs:
            raise KeyError('got an unexpected keyword argument {}'.format(next(iter(kwargs))))


class Affiliation:
  #name
  #city
  #country
  pass
  
