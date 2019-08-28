
class DocumentSet:
    """ Set of documents from retrieved from search query. """

    def __init__(self, docs):
        self.docs = docs
        """ List of documents as `Document` objects. """

    def filter(self, predicate):
        """ Returns a new `DocumentSet` object which includes only documents
        for which the given predicates returns true."""
        return filter(predicate, self.docs)

    def filter_duplicates(self, key=None):
        """ Remove duplicate documents from this `DocumentSet`. The `key`
        lambda is used as identifier to check if two documents are identical.
        By default, equivalence is determined based on DOI (if available)
        or title (if available)."""
        def default_key(document):
            return document.id.id

        if key is None:
            key = default_key

        keys = set()
        result = []

        for doc in self.docs:
            k = key(doc)
            if k not in keys:
                keys.add(k)
                result.append(doc)

        return DocumentSet(result)

    def union(self, other, key=None):
        """ Returns the union of this `DocumentSet` and another `DocumentSet` and
        remove duplicate entries. By default, equivalence is checked using the
        same method as `filter_duplicates`."""
        return DocumentSet(self.docs + other.docs).filter_duplicates(key=key)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, key):
        return self.docs[key]

    def __iter__(self):
        return iter(self.docs)


class DocumentID:
    """The platform transparent ID of a `Document`."""

    def __init__(self, doc_id=None):
        """Initialize the ID of a document."""
        self.id = doc_id

    def parse_scopus(self, scopus_abstract):
        if scopus_abstract.doi is not None:
            self.id = scopus_abstract.doi
        else:
            self.id = scopus_abstract.eid


class Document:
    """ Meta data of academic document. """

    def __init__(self, **kwargs):
        """ Initialize document """

        self.id = kwargs.pop('id')
        """ The platform transparent `DocumentID`. """

        self.title = kwargs.pop('title')
        """ Title of document. """

        self.authors = kwargs.pop('authors', None)
        """ Authors of document as list of `Author` objects, or `None` if unavailable. """

        self.keywords = kwargs.pop('keywords', None)
        """ List of author specified keywords, or `None` if unavailable. """

        self.abstract = kwargs.pop('abstract', None)
        """ Abstract of document, or `None` if unavailable. """

        self.references = kwargs.pop('references', None)
        """List of titles of referenced papers."""

        self.year = kwargs.pop('year', None)
        """ Year of publication as integer. """

        self.source = kwargs.pop('source', None)
        """ Name of source (for example: 
        'The International Conference on Knowledge Discovery and Data Mining') 
        or `None` if unavailable. """

        self.source_type = kwargs.pop('source_type', None)
        """ Type of source (for example:  'Conference Proceedings') or `None` if unavailable. """

        self.citation_count = kwargs.pop('citation_count', None)
        """ The number of received citations, or `None` if unavailable. """

        self._internal = kwargs.pop('internal', None)
        """ Internal object used to extract these properties  """

        self.language = kwargs.pop("language", None)
        """Language of the document, or `None` if unavailable."""

        self.publisher = kwargs.pop("publisher", None)
        """The name of the publisher, or `None` if unavailable."""

        if kwargs:
            raise KeyError('got an unexpected keyword argument {}'.format(next(iter(kwargs))))


class Author:
    """Author of `Document`."""

    def __init__(self, **kwargs):
        self.orcid = kwargs.pop('orcid', None)
        """The ORCID of the author, or `None` if unavailable."""

        self.name = kwargs.pop('name')
        """Name and surname of the author."""

        self.affiliations = kwargs.pop('affiliations', None)
        """Affiliations of the author as list of `Affiliation` objects, or `None` if unavailable."""

        if kwargs:
            raise KeyError('got an unexpected keyword argument {}'.format(next(iter(kwargs))))


class Affiliation:
    """The affiliation of the `Author` of a `Document`."""

    def __init__(self, **kwargs):
        self.name = kwargs.pop("name")
        """The name of the institution."""

        self.city = kwargs.pop("city", None)
        """The city where the institution is, or `None` if unavailable."""

        self.country = kwargs.pop("country", None)
        """The country where the institution is, or `None` if unavailable."""

        if kwargs:
            raise KeyError('got an unexpected keyword argument {}'.format(next(iter(kwargs))))
