from litstudy.sources.ieee import load_ieee_csv
import os

def test_load_ieee_csv():
    path = os.path.dirname(__file__) + '/resources/ieee.csv'
    docs = load_ieee_csv(path)
    doc = docs[0]

    assert doc.title == 'Exascale Computing Trends: Adjusting to the "New Normal"\' for Computer Architecture'
    assert doc.publication_year == 2013
    assert len(doc.keywords) == 37
    assert 'Transistors' in doc.keywords
    assert len(doc.abstract) == 774
    assert doc.citation_count == 51
    assert len(doc.authors) == 2

    author = doc.authors[0]
    assert author.name == 'P. Kogge'
    assert len(author.affiliations) == 1
    assert author.affiliations[0].name == 'University of Notre Dame'
