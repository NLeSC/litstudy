from time import sleep
from typing import Tuple, Optional
from urllib.parse import urlencode, quote_plus
import logging
import requests
import shelve
from ..common import robust_open
import json

from ..common import progress_bar
from ..types import Document, Author, DocumentSet, DocumentIdentifier


def extract_id(item):
    if item is None or not item.get("title"):
        return None

    return DocumentIdentifier(
        item["title"],
        doi=item.get("doi"),
        arxivid=item.get("arxivId"),
        s2id=item.get("paperId"),
        pubmed=item.get("pubmed"),
    )


def extract_ids(items):
    if not items:
        return None

    return list(filter(None, map(extract_id, items)))


class ScholarAuthor(Author):
    def __init__(self, entry):
        self.entry = entry

    @property
    def name(self):
        return self.entry.get("name")

    @property
    def orcid(self):
        return None


class ScholarDocument(Document):
    def __init__(self, entry):
        super().__init__(extract_id(entry))
        self.entry = entry

    @property
    def title(self) -> str:
        return self.entry.get("title")

    @property
    def authors(self):
        authors = self.entry.get("authors")
        if not authors:
            return None

        return [ScholarAuthor(a) for a in authors if a]

    @property
    def publication_year(self):
        return self.entry.get("year")

    @property
    def publication_source(self):
        return self.entry.get("venue")

    @property
    def abstract(self):
        return self.entry.get("abstract")

    @property
    def citations(self):
        return extract_ids(self.entry.get("citations"))

    @property
    def citation_count(self):
        return self.entry.get("numCitedBy")

    @property
    def references(self):
        return extract_ids(self.entry.get("references"))

    def __repr__(self):
        return f"<{self.title}>"

    @staticmethod
    def load(id):
        return fetch_semanticscholar(id)


S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CACHE_FILE = ".semantischolar"
DEFAULT_TIMEOUT = 3.05  # 100 requests per 5 minutes


def request_query(query, offset, limit, cache, session, timeout=DEFAULT_TIMEOUT, extraParams=dict()):
    params=dict(query=query, offset=offset, limit=limit)
    params.update(extraParams)
    encparams = urlencode(params)
    url = f"{S2_QUERY_URL}?{encparams}"

    if url in cache:
        return cache[url]
    sleep(timeout)

    reply = session.get(url,timeout=60*10)
    response = reply.json()

    if "data" not in response:
        msg = response.get("error") or response.get("message") or "unknown"
        if msg.find("Too Many Requests.")>-1 or msg.find("Endpoint request timed out")>-1:
            logging.info(f"request_query: Timeout error while fetching {reply.url}: {msg}")
            return "TIMEOUT"
        else:
            raise Exception(f"error while fetching {reply.url}: {msg}")

    cache[url] = response
    return response


def request_paper(key, cache, session, timeout=DEFAULT_TIMEOUT, extraParams=dict()):
    encparams = urlencode(extraParams)
    url = S2_PAPER_URL + quote_plus(key)+"?"+encparams

    if url in cache:
        return cache[url]

    try:
        sleep(timeout)
        data = session.get(url).json()
    except Exception as e:
        logging.warning(f"failed to retrieve {key}: {e}")
        return None

    if "paperId" not in data:
        msg = data.get("error") or data.get("message") or "unknown error"
        logging.warning(f"failed to retrieve {key}: {msg}")
        return None

    cache[url] = data
    return data


def fetch_semanticscholar(key: set, *, session=None) -> Optional[Document]:
    """Fetch SemanticScholar metadata for the given key. The key can be
    one of the following (see `API reference
    <https://www.semanticscholar.org/product/api>`_):

    * DOI
    * S2 paper ID
    * ArXiv ID (example format: `arXiv:1705.10311`)
    * MAG ID (example format: `MAG:112218234`)
    * ACL ID (example format: `ACL:W12-3903`)
    * PubMed ID (example format: `PMID:19872477`)
    * Corpus ID (example format: `CorpusID:37220927`)

    :param session: The `requests.Session` to use for HTTP requests.
    :returns: The `Document` if it was found and `None` otherwise.
    """

    if key is None:
        return None

    if session is None:
        session = requests.Session()

    with shelve.open(CACHE_FILE) as cache:
        if isinstance(key, DocumentIdentifier):
            data = None
            if data is None and key.s2id:
                data = request_paper(key.s2id, cache, session)

            if data is None and key.doi:
                data = request_paper(key.doi, cache, session)

            if data is None and key.pubmed:
                data = request_paper(f"PMID:{key.pubmed}", cache, session)

            if data is None and key.arxivid:
                data = request_paper(f"arXiv:{key.arxivid}", cache, session)
        else:
            data = request_paper(key, cache, session)

    if data is None:
        return None

    return ScholarDocument(data)


def refine_semanticscholar(docs: DocumentSet, *, session=None) -> Tuple[DocumentSet, DocumentSet]:
    """Attempt to fetch SemanticScholar metadata for each document in the
    given set based on their DOIs. Returns a tuple containing two sets: the
    documents available on SemanticScholar and the remaining documents that
    were not found or do not have a DOI.

    :param session: The `requests.Session` to use for HTTP requests.
    :returns: The documents available on SemanticScholar and the remaining documents.
    """

    def callback(doc):
        if isinstance(doc, ScholarDocument):
            return doc

        return fetch_semanticscholar(doc.id, session=session)

    return docs._refine_docs(callback)


def search_semanticscholar(
    query: str, *, limit: int = None, batch_size: int = 100, session=None
) -> DocumentSet:
    """Submit the given query to SemanticScholar API and return the results
    as a `DocumentSet`. The query is a string containg keywords (see `API reference
    <https://www.semanticscholar.org/product/api%2Ftutorial#using-search-query-operators>`_).

    :param query: The search query to submit.
    :param limit: The maximum number of results to return.
    :param batch_size: The number of results to retrieve per request. Must be at most 100.
    :param session: The `requests.Session` to use for HTTP requests.
    """

    if not query:
        raise Exception("no query specified in `search_semanticscholar`")

    if session is None:
        session = requests.Session()

    docs = []

    with shelve.open(CACHE_FILE) as cache:
        paper_ids = []
        to=0

        while True:
            offset = len(paper_ids)

            response = request_query(query, offset, batch_size, cache, session)
            if not response:
                break
            if response == "TIMEOUT":
                to=to+1
                logging.info("Timeout:",DEFAULT_TIMEOUT*4*to)
                sleep(DEFAULT_TIMEOUT*4*to)
                continue 
            else:
                to=0

            records = response["data"]
            total = response["total"]

            for record in records:
                paper_ids.append(record["paperId"])

            # Check if we reached the total number of papers
            if len(paper_ids) >= total:
                break

            # Check if we exceeded the user-defined limit
            if limit is not None and len(paper_ids) >= limit:
                paper_ids = paper_ids[:limit]
                break

        for paper_id in progress_bar(paper_ids):
            doc = request_paper(paper_id, cache, session)

            if doc:
                docs.append(ScholarDocument(doc))
            else:
                logging.warn(f"could not find paper id {paper_id}")

    return DocumentSet(docs)

def load_semanticscholar_json(path: str) -> DocumentSet:
    """Import json file exported from SemanticScholar"""
    docs = []
    with robust_open(path) as f:
        result = json.load(f)
        data=result["data"]
        for doc in data:
            ids=doc.pop("externalIds")
            for i in ids:
                if i=="DOI":
                    doc["doi"]=ids.get("DOI").lower()
                elif i=="ArXiv":
                    doc["arxivId"]=ids.get("ArXiv")
                elif i=="PubMed":
                    doc["pubmed"]=ids.get("PubMed")
            docs.append(ScholarDocument(doc))
    return DocumentSet(docs)