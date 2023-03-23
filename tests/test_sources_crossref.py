from litstudy.sources.crossref import fetch_crossref, search_crossref
from .common import MockSession


def test_fetch_crossref():
    session = MockSession()
    doc = fetch_crossref("10.1109/SAMOS.2013.6621096", session=session)

    assert doc.title == "Rethinking computer architecture for throughput computing"
    assert doc.id.title == "Rethinking computer architecture for throughput computing"
    assert doc.id.doi == "10.1109/samos.2013.6621096"


def test_search_crossref():
    session = MockSession()
    docs = search_crossref("litstudy", session=session)

    assert any(doc.title == "litstudy: A Python package for literature reviews" for doc in docs)
