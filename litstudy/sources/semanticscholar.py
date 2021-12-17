from .types import Document, Author, DocumentSet, DocumentIdentifier
from urllib.parse import quote_plus
import logging
import requests
import shelve

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it):
        return it

S2_URL = 'http://api.semanticscholar.org/v1/paper/'

def extract_id(item):
    if item is None or not item.get('title'):
        return None

    return DocumentIdentifier(
            item['title'],
            doi=item.get('doi'),
            arxivid=item.get('arxivId'),
            s2id=item.get('paperId'),
    )

def extract_ids(items):
    if not items:
        return None

    return list(filter(None, map(extract_id, items)))


class ScholarAuthor(Document):
    def __init__(self, entry):
        self.entry = entry

    def name(self):
        return self.entry.get('name')

    @property
    def orcid(self):
        return None

    @property
    def affiliations(self) -> 'Optional[list[Affiliation]]':
        return None


class ScholarDocument(Document):
    def __init__(self, entry):
        super().__init__(extract_id(entry))
        self.entry = entry

    @property
    def title(self) -> str:
        return self.entry.get('title')

    @property
    def authors(self):
        authors = self.entry.get('authors')
        if not authors:
            return None

        return [ScholarAuthor(a) for a in authors if a]

    @property
    def publication_year(self):
        return self.entry.get('year')

    @property
    def publication_source(self):
        return self.entry.get('venue')

    @property
    def abstract(self):
        return self.entry.get('abstract')

    @property
    def citations(self):
        return extract_ids(self.entry.get('citations'))

    @property
    def citation_count(self):
        return self.entry.get('numCitedBy')

    @property
    def references(self):
        return extract_ids(self.entry.get('references'))

    def __repr__(self):
        return f'<{self.title}>'

    @staticmethod
    def load(id):
        return search_semanticscholar(id)


CACHE_FILE = '.semantischolar'

def request(key):
    with shelve.open(CACHE_FILE) as cache:
        if key in cache:
            return cache[key]

        url = S2_URL + quote_plus(key)

        try:
            data = requests.get(url).json()
        except Exception as e:
            logging.warn(f'failed to retreive {key}: {msg}')
            return None

        if 'paperId' in data:
            cache[key] = data
            return data
        else:
            msg = data.get('error') or data.get('message') or 'unknown error'
            logging.warn(f'failed to retreive {key}: {msg}')
            return None


def search_semanticscholar(key):
    if isinstance(key, DocumentIdentifier):
        data = None
        if data is None and key.s2id:
            data = request(key.s2id)

        if data is None and key.doi:
            data = request(key.doi)

        if data is None and key.pubmed:
            data = request(f'PMID:{key.pubmed}')

        if data is None and key.arxivid:
            data = request(f'arXiv:{key.arxivid}')
    else:
        data = request(key)

    if data is None:
        return None

    return ScholarDocument(data)


def refine_semanticscholar(originals: DocumentSet) -> DocumentSet:
    docs = []

    for doc in tqdm(originals):
        if not isinstance(doc, ScholarDocument):
            new_doc = search_semanticscholar(doc.id)
            if new_doc is not None:
                doc = new_doc

        docs.append(doc)

    return DocumentSet(docs)

