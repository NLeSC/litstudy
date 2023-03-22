from litstudy.sources.semanticscholar import search_semanticscholar, fetch_semanticscholar


def test_load_s2_file():
    docs = search_semanticscholar('exascale', limit=10)
    assert len(docs) == 10

    docs = search_semanticscholar('litstudy')
    assert any(doc.id.doi == "10.2139/ssrn.4079400" for doc in docs)


def test_fetch_semanticscholar():
    doc = fetch_semanticscholar("arXiv:1705.10311")
    assert doc.title == "Optimal Multi-Object Segmentation with Novel Gradient Vector Flow Based Shape Priors"

    doc = fetch_semanticscholar("MAG:112218234")
    assert doc.title == "Techniques for Measuring Sea Turtles"

    doc = fetch_semanticscholar("ACL:W12-3903")
    assert doc.title == "The Study of Effect of Length in Morphological Segmentation of Agglutinative Languages"

    doc = fetch_semanticscholar("PMID:19872477")
    assert doc.title == "THE COMBINATION OF GELATIN WITH HYDROCHLORIC ACID"
