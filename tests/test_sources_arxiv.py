import pytest
from litstudy.sources.arxiv import search_arxiv


def test_arxiv_query():
    docs = search_arxiv('all:positron',
                        start=0,
                        total_results=200,
                        results_per_iteration=100)

    assert len(docs) == 200
