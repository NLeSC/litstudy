from litstudy.sources.springer import load_springer_csv
import os


def test_load_springer_csv():
    path = os.path.dirname(__file__) + "/resources/springer.csv"
    docs = load_springer_csv(path)
    doc = docs[0]

    assert doc.title == "Graph-Based Load Balancing Model for Exascale Computing Systems"
    assert doc.publication_year == 2022
    assert doc.id.doi == "10.1007/978-3-030-92127-9_33"
