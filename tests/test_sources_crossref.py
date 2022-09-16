from litstudy.sources.crossref import fetch_crossref


def test_fetch_crossref():
    doc = fetch_crossref("10.1109/SAMOS.2013.6621096")

    assert doc.title == "Rethinking computer architecture for throughput computing"
    assert doc.id.title == "Rethinking computer architecture for throughput computing"
    assert doc.id.doi == "10.1109/samos.2013.6621096"
