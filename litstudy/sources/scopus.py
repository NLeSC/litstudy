from .types import Document, DocumentSet, DocumentIdentifier, Author, Affiliation
from pybliometrics.scopus import AbstractRetrieval, ScopusSearch
import itertools
from collections import defaultdict
from datetime import date
import random
import logging

try:
    from tqdm import tqdm
except:
    def tqdm(it):
        return it

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
        return ScopusDocument(AbstractRetrieval(id, id_type=id_type, view=view))

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
                items[aff.indexed_name].append(ScopusAffiliation(aff))

            return [ScopusAuthor(a, f) for a, f in items.items()]

        if self.doc.authors is not None:
            return [ScopusAuthor(a.indexed_name, None) for a in self.doc.authors]

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


def search_scopus(query: str, *, limit:int=None) -> DocumentSet:
    search = ScopusSearch(query, view='STANDARD')
    eids = list(search.get_eids())
    docs = []

    if limit is not None and len(eids) > limit:
        random.seed(0)
        random.shuffle(eids)
        eids = eids[:limit]

    for eid in tqdm(eids):
        doc = ScopusDocument.from_eid(eid)
        docs.append(doc)

    return DocumentSet(docs)

def refine_scopus(originals: DocumentSet) -> DocumentSet:
    docs = []

    for doc in tqdm(originals):
        if not isinstance(doc, ScopusDocument):
            id = doc.id
            doi = id.doi

            if doi is not None:
                try:
                    doc = ScopusDocument.from_doi(doi)
                except Exception as e:
                    logging.warn(f'no document found for DOI {doi}: {e}')

        docs.append(doc)

    return DocumentSet(docs)



