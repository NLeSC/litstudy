

class Document:
  """Example of document"""
  def __init__(self, **kwargs):
    """ Initialize document """

    self.doi = kwargs.get('doi')
    self.title = kwargs.get('title')
    self.authors = kwargs.get('authors')
    self.keywords = kwargs.get('keywords')
    self.abstract = kwargs.get('abstract')
    self.references = kwargs.get('references')
    self.year = kwargs.get('year')
    self.source = kwargs.get('source')

    self.citation_count = kwargs.get('citation_count')
    """ The number of citations """

    self._internal = kwargs.get('internal')
    """ """

class Author:
  def __init__(self, **kwargs):
    self.orcid = kwargs.get('orcid')
    self.name = kwargs.get('name')
    self.affiliations = kwargs.get('affiliations')

class Affiliation:
  #name
  #city
  #country
  pass
  
