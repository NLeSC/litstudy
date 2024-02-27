import os
import pytest
from litstudy.sources.scopus_csv import load_scopus_csv


def test_doc_id_doi_string(doc):
    assert isinstance(doc.id.doi, str) or doc.id.doi is None


def test_doc_title_is_string(doc):
    assert isinstance(doc.title, str)


def test_doc_publication_year_is_int(doc):
    assert isinstance(doc.publication_year, int)


def test_doc_keywords_elements(doc):
    assert all([isinstance(kw, str) for kw in doc.keywords])


def test_doc_authors_elements(doc):
    da = doc.authors
    if da is not None:
        assert all([hasattr(auth, "name") for auth in da])


def test_doc_affiliations_elements(doc):
    da = doc.authors
    if da is not None:
        for auth in doc.authors:
            affs = auth.affiliations
            if affs is not None:
                assert all([hasattr(aff, "name") for aff in affs])


@pytest.fixture
def doc_set_v1():
    path = os.path.dirname(__file__) + "/resources/scopus.csv"
    docs = load_scopus_csv(path)
    return docs


@pytest.fixture
def doc_set_v2():
    path = os.path.dirname(__file__) + "/resources/scopus_v2.csv"
    docs = load_scopus_csv(path)
    return docs


@pytest.fixture
def v1r0(doc_set_v1):
    return doc_set_v1[0]


@pytest.fixture
def v2r0(doc_set_v2):
    return doc_set_v2[0]


@pytest.fixture
def v2r36(doc_set_v2):
    return doc_set_v2[36]


@pytest.fixture
def v1r1(doc_set_v1):
    return doc_set_v1[1]


@pytest.fixture
def v1r6(doc_set_v1):
    return doc_set_v1[6]


def test_v1r0_title(v1r0):
    assert v1r0.title == "Scalable molecular dynamics with NAMD"


def test_v2r0_title(v2r0):
    assert v2r0.title.startswith("Gender-specific visual perturbation effects")


def test_v1r0_abstract(v1r0):
    assert v1r0.abstract.startswith("NAMD is a parallel molecular dynamics code")


def test_v2r0_abstract(v2r0):
    assert v2r0.abstract.startswith(
        "This study investigated the effects of different visual rotation speeds"
    )


def test_v1r0_publication_source(v1r0):
    assert v1r0.publication_source == "Journal of Computational Chemistry"


def test_v2r0_publication_source(v2r0):
    assert v2r0.publication_source == "Ergonomics"


def test_v1r0_language(v1r0):
    assert v1r0.language == "English"


def test_v2r0_language(v2r0):
    assert v2r0.language == "English"


def test_v1r0_publisher(v1r0):
    assert v1r0.publisher == "John Wiley and Sons Inc."


def test_v2r0_publisher(v2r0):
    assert v2r0.publisher == "Taylor and Francis Ltd."


def test_v1r0_citation_count(v1r0):
    assert v1r0.citation_count == 13169


def test_v2r0_citation_count(v2r0):
    assert v2r0.citation_count == 0


def test_v1r0_publication_year(v1r0):
    assert v1r0.publication_year == 2005


def test_v2r0_publication_year(v2r0):
    assert v2r0.publication_year == 2023


def test_v1r0_keywords(v1r0):
    assert v1r0.keywords == [
        "Biomolecular simulation",
        "Molecular dynamics",
        "Parallel computing",
    ]


def test_v1r1_keywords(v1r1):
    assert v1r1.keywords == []


def test_v2r0_keywords(v2r0):
    assert v2r0.keywords == [
        "electromyography",
        "Gait",
        "simulation",
        "space medicine",
        "visual flow",
    ]


def test_v1r0_authors(v1r0):
    assert [auth.name for auth in v1r0.authors] == [
        "Phillips, J.C. (ID: 57202138757)",
        "Braun, R. (ID: 7402220509)",
        "Wang, W. (ID: 56948551400)",
        "Gumbart, J. (ID: 8553717000)",
        "Tajkhorshid, E. (ID: 6701753117)",
        "Villa, E. (ID: 8412476700)",
        "Chipot, C. (ID: 7003715790)",
        "Skeel, R.D. (ID: 7005206020)",
        "Kalé, L. (ID: 7005862685)",
        "Schulten, K. (ID: 7102415947)",
    ]


def test_v1r1_authors(v1r1):
    assert [auth.name for auth in v1r1.authors] == ["Murata T. (ID: 7402736947)"]


def test_v2r0_authors(v2r0):
    assert [auth.name for auth in v2r0.authors] == [
        "Hao J. (ID: 57221302630)",
        "High R. (ID: 6701683718)",
        "Siu K.-C. (ID: 57192938181)",
    ]


def test_v2r0_author_affiliation(v2r0):
    assert [[aff.name for aff in auth.affiliations] for auth in v2r0.authors] == [
        [
            "Department of Health & Rehabilitation Sciences, College of Allied Health Professions, University of Nebraska Medical Center, Omaha, NE, United States"
        ],
        [
            "Department of Biostatistics, College of Public Health, University of Nebraska Medical Center, Omaha, NE, United States"
        ],
        [
            "Department of Health & Rehabilitation Sciences, College of Allied Health Professions, University of Nebraska Medical Center, Omaha, NE, United States"
        ],
    ]


def test_v1r0_author_affiliation(v1r0):
    assert [[aaff.name for aaff in auth.affiliations] for auth in v1r0.authors] == [
        [
            "Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
        [
            "Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
        [
            "Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
        [
            "Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
        [
            "Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
        [
            "Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
        ["UMR CNRS/UHP 7565, Université Henri Poincaré, 54506 Vandaeuvre-les-Nancy, Cedex, France"],
        [
            "Department of Computer Science, Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
        [
            "Department of Computer Science, Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
        [
            "Beckman Institute, University of Illinois at Urbana-Champaign, Urbana, IL 61801, United States"
        ],
    ]


def test_v1r6_author_affiliation(v1r6):
    assert [[aaff.name for aaff in auth.affiliations] for auth in v1r6.authors] == [
        [
            "Department of Human Genetics, And Women's Hospital, Zhejiang University School of Medicine, Hangzhou, Zhejiang, China",
            "Zhejiang Provincial Key Laboratory of Genetic & Developmental Disorders, Zhejiang University School of Medicine, Hangzhou, Zhejiang, China",
        ],
        [
            "Department of Ultrasound, Women's Hospital, Zhejiang University School of Medicine, Hangzhou, Zhejiang, China"
        ],
        [
            "Graduate School of Information Science and Technology, University of Tokyo, Tokyo, Japan"
        ],
        [
            "Department of Human Genetics, And Women's Hospital, Zhejiang University School of Medicine, Hangzhou, Zhejiang, China",
            "Zhejiang Provincial Key Laboratory of Genetic & Developmental Disorders, Zhejiang University School of Medicine, Hangzhou, Zhejiang, China",
        ],
        [
            "Center for Genomic Medicine, Graduate School of Medicine, Kyoto University, Shogoinkawahara-cho, Kyoto-City, Kyoto, Sakyo-ku, Japan"
        ],
        [
            "Laboratory of DNA Information Analysis, Human Genome Center, Institute of Medical Science, University of Tokyo, Tokyo, Japan"
        ],
    ]


def test_v2r36_author_affiliation(v2r36):
    for auth in v2r36.authors:
        assert auth.affiliations == None
