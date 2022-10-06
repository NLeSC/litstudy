from ..common import progress_bar
from datetime import date
from time import sleep
from typing import Tuple, Optional
from urllib.parse import quote_plus, urlencode
import logging
import re
import requests
import shelve

from ..types import Document, Author, DocumentSet, DocumentIdentifier, \
        Affiliation


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
        entries = self.entry.get('affiliation')
        if entries:
            return [CrossRefAffiliation(e) for e in entries]
        else:
            return None


class CrossRefAffiliation(Affiliation):
    def __init__(self, entry):
        self.entry = entry

    @property
    def name(self) -> str:
        return self.entry['name']


def _extract_title(entry):
    # CrossRef returns list of titles from some reason?
    titles = entry.get('title')

    if titles:
        return re.sub(r'[\s]+', ' ', ' '.join(titles))
    else:
        return None


class CrossRefDocument(Document):
    def __init__(self, entry):
        self.entry = entry
        title = _extract_title(entry)
        doi = entry.get('DOI')

        super().__init__(DocumentIdentifier(title, doi=doi))

    @property
    def title(self) -> str:
        return _extract_title(self.entry)

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
        except Exception:
            return None

    @property
    def publication_year(self):
        try:
            return int(self.entry['published-print']['date-parts'][0])
        except Exception:
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
        except Exception:
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
        return fetch_crossref(doi)


CACHE_FILE = '.crossref'
CROSSREF_URL = 'https://api.crossref.org/works/'


def fetch_crossref(doi: str, timeout=0.5) -> Optional[Document]:
    """Fetch the metadata for the given DOI from CrossRef.

    :returns: The `Document` or `None` if the DOI was not available.
    """
    def request():
        if not doi:
            return None

        with shelve.open(CACHE_FILE) as cache:
            if doi in cache:
                return cache[doi]

            url = CROSSREF_URL + quote_plus(doi)

            try:
                response = requests.get(url)
            except Exception as e:
                logging.warn(f'failed to retrieve {doi}: {e}')
                return None

            sleep(timeout)

            code = response.status_code
            if code == 200:
                try:
                    data = response.json()['message']
                except Exception as e:
                    logging.warn(f'invalid output from {url}: {e}')
                    return None
            elif code == 404:
                logging.warn(f'failed to retrieve {doi}: resource not found')
                data = None
            else:
                logging.warn(
                        f'failed to retrieve {doi} ({code}): {response.text}')
                return None

            cache[doi] = data
            return data

    data = request()
    return CrossRefDocument(data) if data else None


def refine_crossref(docs: DocumentSet, timeout=0.5
                    ) -> Tuple[DocumentSet, DocumentSet]:
    """Attempts to fetch metadata from CrossRef for each document in the given
    set. Returns a tuple of two sets: the documents retrieved from CrossRef
    and the remaining documents (i.e., without DOI or not found).

    :param timeout: Timeout in seconds between each request to throttle
        server communication.
    """
    def callback(doc):
        if isinstance(doc, CrossRefDocument):
            return doc

        return fetch_crossref(doc.id.doi, timeout)

    return docs._refine_docs(callback)


def _fetch_dois(params: dict, timeout: float, limit: int):
    dois = []

    params = dict(params)
    params["cursor"] = "*"  # The cursor should be * on first request
    params["rows"] = 100  # Fetch 100 results per request

    while True:
        query_string = urlencode(params)
        url = CROSSREF_URL + "?" + query_string

        try:
            response = requests.get(url).json()
        except Exception as e:
            logging.warn(f'failed to retrieve {url}: {e}')
            return None

        # Status should be "ok"
        if response["status"] != "ok":
            raise ValueError(f"failed to retrieve {url}: {response}")

        sleep(timeout)

        data = response["message"]
        items = data["items"]
        params["cursor"] = data["next-cursor"]

        # If no more items are found, break from loop
        if not items:
            break

        # Append to dois
        for item in items:
            doi = item["DOI"]
            dois.append(doi)

        # If hit limit, break from loop
        if limit and len(dois) > limit:
            dois = dois[:limit]
            break

    return dois


def search_crossref(query: str, limit: int = None, timeout: float = 0.5,
                    options: dict = dict()) -> DocumentSet:
    """ Submit the query to the CrossRef API.

    :param query: The search query.
    :param limit: Maximum number of results to retrieve. ``None`` is unlimited.
    :param timeout: Timeout in seconds between each request to throttle
                    server communication
    :param options: Additional parameters that are passed to the ``/works``
                    endpoint of CrossRef (see `CrossRef API`
                    <https://api.crossref.org>`_). Options are `sort` and
                    `filter`.
    """
    if not query:
        return DocumentSet()

    params = dict()
    params["query"] = query
    params["select"] = "DOI"

    for key, value in options.items():
        params[key] = value

    cache_params = dict(params)
    cache_params["limit"] = limit
    cache_key = urlencode(cache_params)

    with shelve.open(CACHE_FILE) as cache:
        if cache_key not in cache:
            dois = _fetch_dois(params, timeout, limit)
            cache[cache_key] = dois
        else:
            dois = cache[cache_key]

    docs = []
    for doi in progress_bar(dois):
        docs.append(fetch_crossref(doi))

    return DocumentSet(docs)
