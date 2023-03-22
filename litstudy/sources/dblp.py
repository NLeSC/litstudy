from ..types import Document, DocumentSet, DocumentIdentifier, Author
import requests
import shelve
import logging


class DBLPDocument(Document):
    def __init__(self, entry, authors):
        id = DocumentIdentifier(entry["title"], doi=entry.get("doi"))
        super().__init__(id)

        self.entry = entry
        self._authors = authors

    @property
    def title(self):
        return self.entry.get("title")

    @property
    def publication_year(self):
        try:
            return int(self.entry.get("year"))
        except Exception:
            return None

    @property
    def publication_type(self):
        return self.entry.get("type")

    @property
    def publication_source(self):
        return self.entry.get("venue")

    @property
    def publisher(self):
        return self.entry.get("publisher")

    @property
    def authors(self):
        return self._authors

    def __repr__(self):
        return f"<{self.title}>"


class DBLPAuthor(Author):
    def __init__(self, pid, name):
        self._pid = pid
        self._name = name

    @property
    def pid(self):
        return self._pid

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"<{self.name}>"


def process_authors(entry, author_cache):
    # Sometimes, authors is not a valid key
    if "authors" not in entry:
        return None

    inputs = entry["authors"]["author"]
    outputs = []

    # Sometimes, inputs is nothing? (empty str or null)
    if not inputs:
        return None

    # Sometimes, inputs is str? (single author name)
    if isinstance(inputs, str):
        return [DBLPAuthor(None, inputs)]

    # Sometimes, input is dict? (single-item list)
    if isinstance(inputs, dict):
        inputs = [inputs]

    # Sometimes inputs is list
    for author in inputs:
        pid = author["@pid"]
        name = author["text"]

        if pid not in author_cache:
            author_cache[pid] = DBLPAuthor(pid, name)

        outputs.append(author_cache[pid])

    return outputs


CACHE_FILE = ".dblp"
DBLP_URL = "http://dblp.org/search/publ/api"


def search_dblp(query: str, *, limit=None) -> DocumentSet:
    """Perform the given `query` on the DBLP API and return the results
    as a `DocumentSet`.

    :param limit: The maximum number of documents to retrieve.
    """

    attr = dict(format="json", h=100, q=query, f=0)
    offset = 0

    docs = []
    author_cache = dict()

    with shelve.open(CACHE_FILE) as cache:
        while True:
            key = f"{query};{offset}"

            if key not in cache:
                attr["f"] = offset
                req = requests.get(DBLP_URL, params=attr)
                cache[key] = req.json().get("result")

            data = cache[key]

            if not data:
                break

            status = data.get("status").get("text")
            if status != "OK":
                logging.warning(f"expecting status OK, got status {status}")
                break

            if "hits" not in data or "hit" not in data["hits"]:
                break

            entries = data["hits"]["hit"]
            offset += len(entries)

            if not entries:
                break

            for entry in entries:
                entry = entry["info"]

                authors = process_authors(entry, author_cache)
                docs.append(DBLPDocument(entry, authors))

            if limit is not None and len(docs) >= limit:
                break

    return DocumentSet(docs)
