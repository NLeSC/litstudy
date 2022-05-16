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
        return self.entry.get('arxiv_doi', None)

    @property
    def title(self) -> str:
        return self.entry.get('title')

    @property
    def authors(self) -> List:
        return [ArXivAuthor(name.get('name'))
                for name
                in self.entry.get('authors')]

    @property
    def journal_ref(self) -> Optional[str]:
        return self.entry.get('arxiv_journal_ref', None)

    @property
    def publication_date(self):
        publication_date = datetime.strptime(self.entry.get('published'),
                                             "%Y-%m-%dT%H:%M:%SZ")
        return publication_date.date()

    @property
    def abstract(self) -> Optional[str]:
        return self.entry.get('summary', None)

    @property
    def language(self) -> Optional[str]:
        return self.entry.get('language', None)

    @property
    def category(self) -> Optional[List[str]]:
        '''returns arxiv category for article'''
        return self.entry.get('tags', None)[0].get('term', None)

# Base api query url
ARXIV_SEARCH_URL = 'http://export.arxiv.org/api/query'

def search_arxiv(search_query,
                 start=0,
                 total_results=100,
                 results_per_iteration=100,
                 sleep_time=3) -> DocumentSet:

    '''
    Search parameters:
    ----------------------------------------------------------------
    search_query = all:electron    #search for electron in all fields
    start                          #start at the first result
    total_results                  #total results
    results_per_iteration          #results at a time
    sleep_time                     #number of seconds to wait beetween calls
    ----------------------------------------------------------------
    Returns(DocumentSet with a folowing parameters)
    ----------------------------------------------------------------
    authors: list of authors separated by commas
    title: title of the article
    abstract: abstract of the article
    published: publish_date in datetime format
    arxiv_journal_ref: reference to the journal if existed
    doi: digital document identifier
    keywords: terms relevent to the document
    language: language document is written in '''

    docs = list()

    for i in range(start, total_results, results_per_iteration):
        query = urlencode(dict(
            search_query=search_query,
            start=i,
            max_results=results_per_iteration
        ))

        url = f'{ARXIV_SEARCH_URL}?{query}'
        data = feedparser.parse(url)

        for entry in data.entries:
            docs.append(ArXivDocument(entry))

        time.sleep(sleep_time)

    return DocumentSet(docs)
