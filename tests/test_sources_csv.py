from litstudy import load_csv
import os


def test_load_ieee_csv():
    path = os.path.dirname(__file__) + '/resources/ieee.csv'
    docs = load_csv(path)
    doc = docs[0]

    assert doc.title == 'Exascale Computing Trends: Adjusting to the "New Normal"\' for Computer Architecture'
    assert doc.publication_year == 2013
    #assert len(doc.keywords) == 37
    assert 'Transistors' in doc.keywords
    assert len(doc.abstract) == 774
    assert doc.citation_count == 51
    assert len(doc.authors) == 2

    author = doc.authors[0]
    assert author.name == 'P. Kogge'


def test_load_springer_csv():
    path = os.path.dirname(__file__) + '/resources/springer.csv'
    docs = load_csv(path)
    doc = docs[0]

    assert doc.title == 'Graph-Based Load Balancing Model for Exascale Computing Systems'
    assert doc.publication_year == 2022
    assert doc.id.doi == '10.1007/978-3-030-92127-9_33'


def test_load_scopus_csv():
    path = os.path.dirname(__file__) + '/resources/scopus.csv'
    docs = load_csv(path)
    doc = docs[0]

    assert doc.title == "Scalable molecular dynamics with NAMD"
    assert doc.abstract.startswith("NAMD is a parallel molecular dynamics code")
    assert doc.publication_source == "Journal of Computational Chemistry"
    assert doc.language is None
    assert doc.publisher == "John Wiley and Sons Inc."
    assert doc.citation_count == 13169
    assert doc.keywords == ['Biomolecular simulation', 'Molecular dynamics', 'Parallel computing']
    assert doc.publication_year == 2005

    assert len(doc.authors) == 10
    assert doc.authors[0].name == "Phillips J.C."
