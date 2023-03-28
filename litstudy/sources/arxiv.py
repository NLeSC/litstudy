from litstudy.types import Document, DocumentSet, DocumentIdentifier, Author
from typing import Optional, List
import feedparser  # type: ignore
from datetime import datetime
from urllib.parse import urlencode
import time


class ArXivAuthor(Author):
    def __init__(self, entry):
        self.entry = entry

    @property
    def name(self):
        return self.entry


class ArXivDocument(Document):
    def __init__(self, entry):
        identifier = DocumentIdentifier(
            entry.title,
        )

        super().__init__(identifier)
        self.entry = entry

    @property
    def doi(self) -> Optional[str]:
        return self.entry.get("arxiv_doi", None)

    @property
    def title(self) -> str:
        return self.entry.get("title")

    @property
    def authors(self) -> List:
        return [ArXivAuthor(name.get("name")) for name in self.entry.get("authors")]

    @property
    def journal_ref(self) -> Optional[str]:
        return self.entry.get("arxiv_journal_ref", None)

    @property
    def publication_date(self):
        publication_date = datetime.strptime(self.entry.get("published"), "%Y-%m-%dT%H:%M:%SZ")
        return publication_date.date()

    @property
    def abstract(self) -> Optional[str]:
        return self.entry.get("summary", None)

    @property
    def language(self) -> Optional[str]:
        return self.entry.get("language", None)

    @property
    def category(self) -> Optional[List[str]]:
        """returns arxiv category for article"""
        return self.entry.get("tags", None)[0].get("term", None)


# Base api query url
ARXIV_SEARCH_URL = "http://export.arxiv.org/api/query"


def search_arxiv(
    query,
    start=0,
    max_results=2000,
    batch_size=100,
    sort_order="descending",
    sort_by="submittedDate",
    sleep_time=3,
) -> DocumentSet:
    """Search `arXiv <https://arxiv.org/>`_.

    Each returned document contains the following attributes:
    title, authors, doi, journal_ref, publication_date, abstract, language, and category

    :param query: The query as described in the
                  `arXiv API use manual <https://info.arxiv.org/help/api/user-manual.html#query_details>`_.
    :param max_results: The maximum number of results to return.
    :param start: Skip the first ``start`` documents from the results.
    :param batch_size: The number documents to fetch per request.
    :param sleep_time: The time to wait in seconds between each HTTP requests.
    """

    docs = list()
    start = int(start)
    max_results = int(max_results)
    batch_size = int(batch_size)

    while len(docs) < max_results:
        url_query = urlencode(
            dict(
                search_query=query,
                start=start,
                max_results=min(max_results - len(docs), batch_size),
                sortOrder=sort_order,
                sortBy=sort_by,
            )
        )

        url = f"{ARXIV_SEARCH_URL}?{url_query}"
        data = feedparser.parse(url)

        if not data.entries:
            break

        start += len(data.entries)

        for entry in data.entries:
            docs.append(ArXivDocument(entry))

        time.sleep(sleep_time)

    return DocumentSet(docs)
