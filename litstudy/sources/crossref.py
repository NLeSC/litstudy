from ..common import progress_bar
from .types import Document, Author, DocumentSet, DocumentIdentifier
from datetime import date
from time import sleep
from typing import Tuple
from urllib.parse import quote_plus
import logging
import re
import requests
import shelve


class CrossRefAuthor(Author):
    def __init__(self, entry):
        self.entry = entry

    @property
    def name(self):
        given = self.entry.get('given', '')
        family = self.entry.get('family', '')

        if not given and not family:
            return None

        return f'{given} {family}'.strip()

    @property
    def orcid(self):
        return self.entry.get('ORCID')

    @property
    def affiliations(self):
        return None  # TODO


class CrossRefDocument(Document):
    def __init__(self, entry):
        title = entry.get('title')
        doi = entry.get('DOI')

        super().__init__(DocumentIdentifier(title, doi=doi))
        self.entry = entry

    @property
    def title(self) -> str:
        title = self.entry.get('title')
        if title:
            return re.sub(r'[\s]+', ' ', ' '.join(title))
        else:
            return None

    @property
    def authors(self):
        authors = self.entry.get('author', [])
        return [CrossRefAuthor(a) for a in authors]

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
        try:
            return int(self.entry['published-print']['date-parts'][0])
        except Exception as e:
            return None

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
            return int(self.entry['is-referenced-by-count'])
        except Exception as e:
            return None

    @property
    def references(self):
        output = []

        for ref in self.entry.get('reference', []):
            title = ref.get('unstructured')
            doi = ref.get('DOI')

            if title or doi:
                output.append(DocumentIdentifier(title, doi=doi))

        return output

    def __repr__(self):
        return f'<{self.title}>'

    @staticmethod
    def load(doi):
        return search_crossref(doi)


CACHE_FILE = '.crossref'
CROSSREF_URL = 'https://api.crossref.org/works/'


def request(doi):
    if not doi:
        return False, None

    with shelve.open(CACHE_FILE) as cache:
        if doi in cache:
            return False, cache[doi]

        url = CROSSREF_URL + quote_plus(doi)

        try:
            response = requests.get(url)
        except Exception as e:
            logging.warn(f'failed to retrieve {doi}: {e}')
            return True, None

        code = response.status_code
        if code == 200:
            try:
                data = response.json()['message']
            except Exception as e:
                logging.warn(f'invalid output from {url}: {e}')
                return True, None
        elif code == 404:
            logging.warn(f'failed to retrieve {doi}: resource not found')
            data = None
        else:
            logging.warn(f'failed to retrieve {doi} ({code}): {response.text}')
            return True, None

        cache[doi] = data
        return True, data


def search_crossref(doi):
    is_fresh, data = request(doi)  # include timeout?
    return CrossRefDocument(data) if data else None


def refine_crossref(originals: DocumentSet, timeout=0.5
                    ) -> Tuple[DocumentSet, DocumentSet]:
    original = []
    replaced = []

    for doc in progress_bar(originals):
        if not isinstance(doc, CrossRefDocument):
            is_fresh, data = request(doc.id.doi)

            if is_fresh:  # Fresh request
                sleep(timeout)

            if data:
                replaced.append(CrossRefDocument(data))
                continue

        original.append(doc)

    return replaced, original
