import os
from litstudy.sources.scopus_csv import load_scopus_csv

#def test_doc_title_is_string(doc):
#    assert isinstance(doc.title, str)

#def test_doc_publication_year_is_int(doc):
#    assert isinstance(doc.publication_year, int)

#def test_doc_keywords_is_list_or_none(doc):
#    assert isinstance(doc.keywords, list) or doc.keywords is None

#def test_doc_authors(doc):
#    authors = doc.author

def test_load_scopus_csv_v2():
    path = os.path.dirname(__file__) + "/resources/scopus_v2.csv"
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
            assert doc.title.startswith("Gender-specific visual perturbation effects")
            assert doc.abstract.startswith("This study investigated the effects of different visual rotation speeds")
            assert doc.publication_source == "Ergonomics"
            assert doc.language == "English"
            assert doc.publisher == "Taylor and Francis Ltd."
            assert doc.citation_count == 0
            assert doc.keywords == [
                'electromyography',
                'Gait',
                'simulation',
                'space medicine',
                'visual flow']
            assert doc.publication_year == 2023
            assert len(doc.authors) == 3
            assert doc.authors[0].name == "Hao J. (ID: 57221302630)"
            assert doc.authors[0].affiliations[0].name == "Department of Health & Rehabilitation Sciences, College of Allied Health Professions, University of Nebraska Medical Center, Omaha, NE, United States"

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
            assert doc.authors[0].affiliations[0].name == "Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
