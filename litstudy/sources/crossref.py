from .types import Document, Author, DocumentSet, DocumentIdentifier
from urllib.parse import quote_plus
from datetime import date
import logging
import requests
import shelve

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it):
        return it

CROSSREF_URL = 'https://api.crossref.org/works/'


class CrossRefDocument(Document):
    def __init__(self, entry):
        todo()

    @property
    def title(self) -> str:
        title = self.entry.get('title')
        if title:
            return title[0]
        else:
            return None

    @property
    def authors(self):
        pass

    @property
    def publisher(self):
        return self.entry.get('publisher')

    @property
    def language(self):
        return self.get('language')

    @property
    def publication_date(self):
        try:
            parts = self.entry['published-print']['date-parts']
            year = int(parts[0])
            month = int(parts[1])
            return date(year, month, 1)
        except Exception as e:
            return None

    @property
    def publication_year(self):
        date = self.publication_date
        if date is None:
            return None
        return date.year

    @property
    def publication_source(self):
        source = self.entry.get('container-title')
        if source:
            return source[0]
        else:
            return None

    @property
    def abstract(self):
        return self.entry.get('abstract')

    @property
    def citation_count(self):
        try:
            return int(self.entry.get('is-referenced-by-count'))
        except Exception as e:
            return None

    @property
    def references(self):
        return None

    def __repr__(self):
        return f'<{self.title}>'

    @staticmethod
    def load(doi):
        return search_crossref(doi)


CACHE_FILE = '.crossref'

def request(doi):
    with shelve.open(CACHE_FILE) as cache:
        url = CROSSREF_URL + quote_plus(doi)
        print(url)
        if doi in cache:
            return cache[doi]

        from pprint import pprint

        try:
            data = requests.get(url).json()['message']
        except Exception as e:
            logging.warn(f'failed to retreive {key}: {msg}')
            return None

        cache[doi] = data
        return data


def search_crossref(doi):
    if not doi:
        return None

    data = request(doi)

    if not data:
        return None

    return CrossRefDocument(data)


def refine_crossref(originals: DocumentSet) -> DocumentSet:
    docs = []

    for doc in tqdm(originals):
        if not isinstance(doc, CrossRefDocument):
            new_doc = search_crossref(doc.id.doi)

            if new_doc is not None:
                doc = new_doc

        docs.append(doc)

    return DocumentSet(docs)

