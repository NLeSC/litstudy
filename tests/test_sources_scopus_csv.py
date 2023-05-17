from litstudy.sources.scopus_csv import load_scopus_csv
import os


def test_load_scopus_csv():
    path = os.path.dirname(__file__) + "/resources/scopus.csv"
    docs = load_scopus_csv(path)
    for num, doc in enumerate(docs):
        title = doc.title
        doc_id_title = doc.id.title
        doc_id_doi = doc.id.doi
        doc_id_pubmed = doc.id.pubmed
        doc_id_scopus = doc.id.scopusid
        pub_year = doc.publication_year
        keywords = doc.keywords
        abstract = doc.abstract
        citation_count = doc.citation_count
        publication_source = doc.publication_source
        source_type = doc.source_type
        for author in doc.authors:
            author_name = author.name
            for aff in author.affiliations:
                affiliation = aff.name
        if num == 0:
            assert title == doc_id_title
            assert doc.title == "Scalable molecular dynamics with NAMD"
            assert doc.abstract.startswith("NAMD is a parallel molecular dynamics code")
            assert doc.publication_source == "Journal of Computational Chemistry"
            assert doc.language == "English"
            assert doc.publisher == "John Wiley and Sons Inc."
            assert doc.citation_count == 13169
            assert doc.keywords == [
                "Biomolecular simulation",
                "Molecular dynamics",
                "Parallel computing",
            ]
            assert doc.publication_year == 2005

            assert len(doc.authors) == 10
            assert doc.authors[0].name == "Phillips, J.C. (ID: 57202138757)"
