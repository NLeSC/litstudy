from litstudy.types import Document, DocumentSet
from typing import Optional, List
import feedparser  # type: ignore
from datetime import datetime
import time


class ArXivDocument(Document):
    def __init__(self, entry) -> None:
        self.entry = entry

    @property
    def doi(self) -> Optional[str]:
        return self.entry.get('arxiv_doi', None)

    @property
    def title(self) -> str:
        return self.entry.get('title')

    @property
    def authors(self) -> List:
        return [name.get('name') for name in self.entry.get('authors')]

    @property
    def journal_ref(self) -> Optional[str]:
        return self.entry.get('arxiv_journal_ref', None)

    @property
    def publication_date(self):
        publication_date = datetime.strptime(self.entry.get('published'),
                                             "%Y-%m-%dT%H:%M:%SZ")
        return publication_date.date()


def arxiv_query(search_query,
                start=0,
                total_results=100,
                results_per_iteration=100) -> DocumentSet:
    '''
    Search parameters:
    ----------------------------------------------------------------
    search_query = all:electron    #search for electron in all fields
    start                          #start at the first result
    total_results                  #total results
    results_per_iteration          #results at a time
    ----------------------------------------------------------------
    Returns(list of dictionary with a following keys)
    ----------------------------------------------------------------
    main_author: main_author of the article
    authors: list of authors separated by commas
    url: url of the article
    pdf_url: pdf url of the article
    title: title of the article
    abstract: abstract of the article
    published: publish_date in datetime format
    comment: comment of the article if available
    arxiv_journal_ref: reference to the journal if existed'''

    docs = list()

    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?'

    print(f'Searching arXiv for {search_query}')

    for i in range(start, total_results, results_per_iteration):
        query = (f'search_query={search_query}&start={i}&max_results='
                 f'{results_per_iteration}')

        url = base_url + query
        data = feedparser.parse(url)
        [docs.append(ArXivDocument(entry)) for entry in data.entries]
        # sleeping before api calls (recomendation is 3s)
        time.sleep(3)

    return DocumentSet(docs)
