from litstudy.sources.arxiv import arxiv_query


def test_arxiv_query():
    docs = arxiv_query('all:positron',
                       start=0,
                       total_results=200,
                       results_per_iteration=100)

    assert len(docs) == 200
    d1 = docs[0]

    assert d1.title == ('Positron Transport And Annihilation '
                        'In The Galactic Bulge')
    assert d1.authors == ['Fiona H. Panther']
