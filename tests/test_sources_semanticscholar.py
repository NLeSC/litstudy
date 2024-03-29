from litstudy.sources.semanticscholar import (
    search_semanticscholar,
    fetch_semanticscholar,
    refine_semanticscholar,
)
from .common import MockSession, example_docs


def test_load_s2_file():
    session = MockSession()

    docs = search_semanticscholar("exascale", limit=3, session=session)
    assert len(docs) == 3

    docs = search_semanticscholar("litstudy", session=session)
    assert any(doc.id.doi == "10.2139/ssrn.4079400" for doc in docs)


def test_fetch_semanticscholar():
    session = MockSession()

    doc = fetch_semanticscholar("arXiv:1705.10311", session=session)
    assert (
        doc.title
        == "Optimal Multi-Object Segmentation with Novel Gradient Vector Flow Based Shape Priors"
    )

    doc = fetch_semanticscholar("MAG:112218234", session=session)
    assert doc.title == "Techniques for Measuring Sea Turtles"

    doc = fetch_semanticscholar("ACL:W12-3903", session=session)
    assert (
        doc.title
        == "The Study of Effect of Length in Morphological Segmentation of Agglutinative Languages"
    )

    doc = fetch_semanticscholar("PMID:19872477", session=session)
    assert doc.title == "THE COMBINATION OF GELATIN WITH HYDROCHLORIC ACID"


def test_refine_semanticscholar():
    session = MockSession()

    docs = example_docs()
    found, notfound = refine_semanticscholar(docs, session=session)

    assert len(found) == 1
    assert found[0].title == docs[0].title

    assert len(notfound) == 1
    assert notfound[0].title == docs[1].title
