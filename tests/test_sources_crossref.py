from litstudy.sources.crossref import fetch_crossref, search_crossref


def test_fetch_crossref():
    doc = fetch_crossref("10.1109/SAMOS.2013.6621096")

    assert doc.title == "Rethinking computer architecture for throughput computing"
    assert doc.id.title == "Rethinking computer architecture for throughput computing"
    assert doc.id.doi == "10.1109/samos.2013.6621096"


def test_search_crossref():
    docs = search_crossref("litstudy")

    assert any(doc.title == "litstudy: A Python package for literature reviews" for doc in docs)
