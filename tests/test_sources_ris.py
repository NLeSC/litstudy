from litstudy.sources.ris import load_ris_file
import os


def test_load_ris_file():
    path = os.path.dirname(__file__) + '/resources/example.ris'
    docs = load_ris_file(path)
    doc = docs[0]

    assert doc.title == 'The European Approach to the Exascale Challenge'
    assert doc.publication_year == 2019
    assert len(doc.keywords) == 6
    assert 'Ecosystems' in doc.keywords
    assert len(doc.abstract) == 990
    assert len(doc.authors) == 2

    author = doc.authors[0]
    assert author.name == 'G. Kalbe'
