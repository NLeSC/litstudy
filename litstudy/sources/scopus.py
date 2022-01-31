from ..common import progress_bar, canonical
from ..types import Document, DocumentSet, DocumentIdentifier, Author, \
                   Affiliation
from collections import defaultdict
from datetime import date
from pybliometrics.scopus import AbstractRetrieval, ScopusSearch
from pybliometrics.scopus.exception import Scopus404Error
from typing import Tuple
import logging
import random
import shelve


SCOPUS_CACHE = '.scopus'


class ScopusAuthor(Author):
    def __init__(self, name, affiliations):
        self._name = name
        self._affiliations = affiliations

    @property
    def name(self):
        return self._name

    @property
    def affiliations(self):
        return self._affiliations


class ScopusAffiliation(Affiliation):
    def __init__(self, affiliation):
        self._affiliation = affiliation

    @property
    def name(self) -> str:
        return self._affiliation.organization

    @property
    def country(self):
        return self._affiliation.country or None

    def __repr__(self):
        return f'<{self.name}>'


class ScopusDocument(Document):
    @staticmethod
    def from_identifier(id, id_type, view='FULL'):
        with shelve.open(SCOPUS_CACHE) as cache:
            key = id + '_found'
            if cache.get(key) is False:
                raise Scopus404Error()

            try:
                result = AbstractRetrieval(id, id_type=id_type, view=view)
                return ScopusDocument(result)
            except Scopus404Error:
                cache[key] = False
                raise

    @staticmethod
    def from_eid(eid, **kwargs):
        return ScopusDocument.from_identifier(eid, 'eid', **kwargs)

    @staticmethod
    def from_doi(doi, **kwargs):
        return ScopusDocument.from_identifier(doi, 'doi', **kwargs)

    def __init__(self, doc):
        identifier = DocumentIdentifier(
                doc.title,
                doi=doc.doi,
                isbn=doc.isbn,
                pubmed=doc.pubmed_id,
                eid=doc.eid,
        )

        super().__init__(identifier)
        self.doc = doc

    @property
    def title(self):
        return self.doc.title or None

    @property
    def authors(self):
        if self.doc.authorgroup is not None:
            items = defaultdict(list)

            for aff in self.doc.authorgroup:
                name = f'{aff.indexed_name} (AUID: {aff.auid})'
                items[name].append(ScopusAffiliation(aff))

            return [ScopusAuthor(a, f) for a, f in items.items()]

        if self.doc.authors is not None:
            at = self.doc.authors
            return [ScopusAuthor(a.indexed_name, None) for a in at]

        return None

    @property
    def publisher(self):
        return self.doc.publisher or None

    @property
    def language(self):
        return self.doc.language or None

    @property
    def keywords(self):
        return self.doc.authkeywords or []

    @property
    def abstract(self):
        return self.doc.abstract or self.doc.description or None

    @property
    def citation_count(self):
        if self.doc.citedby_count is not None:
            return int(self.doc.citedby_count)
        return None

    @property
    def references(self):
        refs = []

        if not self.doc.references:
            return None

        for ref in self.doc.references:
            refs.append(DocumentIdentifier(
                ref.title,
                eid=ref.id,
                doi=ref.doi,
            ))

        return refs

    @property
    def publication_source(self):
        return self.doc.confname or self.doc.publicationName or None

    @property
    def source_type(self):
        return self.doc.aggregationType

    @property
    def publication_date(self):
        if self.doc.confdate:
            year, month, day = self.doc.confdate[0]
            return date(year, month, day)

        if self.doc.coverDate:
            try:
                year, month, day = self.doc.coverDate.split('-')
                return date(int(year), int(month), int(day))
            except Exception:
                pass

        return None

    def __repr__(self):
        return f'<{self.title}>'


def fetch_scopus(key: str) -> Optional[Document]:
    """ Fetch the document on Scopus for the given key. The key can be one of
    the following options:

    * DOI
    * Scopus EID or Scopus ID
    * PII
    * Pubmed-ID
    """
    return ScopusDocument.from_identifier(key, None)


def search_scopus(query: str, *, limit: int = None) -> DocumentSet:
    """ Submit the given query to the Scopus API.

    :param limit: Restrict results the first `limit` documents.
    """

    search = ScopusSearch(query, view='STANDARD')
    eids = list(search.get_eids())
    docs = []

    if limit is not None and len(eids) > limit:
        random.seed(0)
        random.shuffle(eids)
        eids = eids[:limit]

    for eid in progress_bar(eids):
        doc = ScopusDocument.from_eid(eid)
        docs.append(doc)

    return DocumentSet(docs)


def refine_scopus(docs: DocumentSet, *, search_title=True
                  ) -> Tuple[DocumentSet, DocumentSet]:
    """Attempt to fetch Scopus metadata for each document in the given
    set. Returns a tuple containing two sets: the documents available on
    Scopus and the remaining documents not found on Scopus.


    Documents are retrieved based on their identifier (DOI, Pubmed ID, or
    Scopus ID). Documents without a unique identifier are retrieved by
    performing a fuzzy search based on their title. This is not ideal
    and can lead to false positives (i.e., another document is found having
    the same title), thus it can be disabled if necessary.

    :param search_title: Flag to toggle searching by title."""
    def callback(doc):
        id = doc.id
        if isinstance(doc, ScopusDocument):
            return doc

        if doi := id.doi:
            try:
                return ScopusDocument.from_doi(doi)
            except Exception as e:
                logging.warn(f'no document found for DOI {doi}: {e}')
                return None

        title = canonical(id.title)
        if len(title) > 10 and search_title:
            query = f'TITLE({title})'
            response = ScopusSearch(query, view='STANDARD', download=False)
            nresults = response.get_results_size()

            if nresults > 0 and nresults < 10:
                response = ScopusSearch(query, view='STANDARD')

                for record in response.results or []:
                    if canonical(record.title) == title:
                        return ScopusDocument.from_eid(record.eid)

        return None

    return docs._refine_docs(callback)
