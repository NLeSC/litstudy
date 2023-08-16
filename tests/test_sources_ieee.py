from litstudy.sources.ieee import load_ieee_csv
import os


def test_load_ieee_csv():
    path = os.path.dirname(__file__) + "/resources/ieee.csv"
    docs = load_ieee_csv(path)
    doc = docs[0]

    assert (
        doc.title
        == 'Exascale Computing Trends: Adjusting to the "New Normal"\' for Computer Architecture'
    )
    assert doc.publication_year == 2013
    assert len(doc.keywords) == 37
    assert "Transistors" in doc.keywords
    assert len(doc.abstract) == 774
    assert doc.citation_count == 51
    assert len(doc.authors) == 2

    author = doc.authors[0]
    assert author.name == "P. Kogge"
    assert len(author.affiliations) == 1
    assert author.affiliations[0].name == "University of Notre Dame"

    # For the second document, the number of authors does not match the number of
    # affiliations. This means we can get the affiliations via `doc.affiliations`,
    # but the authors do not have affilations themselves
    doc = docs[1]
    assert doc.title == "European HPC Landscape"
    assert len(doc.affiliations) == 7
    assert len(doc.authors) == 6
    assert doc.authors[0].affiliations is None
