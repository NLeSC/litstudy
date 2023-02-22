from litstudy.sources.arxiv import search_arxiv


def test_arxiv_query():
    docs = search_arxiv('all:positron',
                        start=0,
                        max_results=10,
                        batch_size=3)

    assert len(docs) == 10
