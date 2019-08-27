from pybliometrics.scopus import ScopusSearch, AbstractRetrieval, AuthorRetrieval
from pybliometrics.scopus.exception import ScopusQueryError

from .common import Document, DocumentID, DocumentSet, Author, Affiliation


def search_mockup():
    a = Document(
        title=' A unified analytical theory of heteropolymers for sequence-specific phase behaviors of polyelectrolytes and polyampholytes ',
        authors=[
            Author(name='Yi-Hsuan Lin'),
            Author(name='Jacob P. Brady'),
            Author(name='Hue Sun Chan'),
            Author(name='Kingshuk Ghosh'),
        ],
        doi='arXiv:1908.09726',
        year=2019)

    b = Document(
        title='Scaling methods for accelerating kinetic Monte Carlo simulations of chemical reaction networks',
        authors=[
            Author(name='Yen Ting Lin'),
            Author(name='Song Feng'),
            Author(name='William S. Hlavacek'),
        ],
        year=2019)

    c = Document(
        title='Discreteness of populations enervates biodiversity in evolution',
        authors=[
            Author(name='Yen-Chih Lin'),
            Author(name='Tzay-Ming Hong'),
            Author(name='Hsiu-Hau Lin'),
        ],
        doi='arXiv:1005.4335',
        year=2010)

    d = Document(
        title='Phylogenetic Analysis of Cell Types using Histone Modifications',
        authors=[
            Author(name='Nishanth Ulhas Nair'),
            Author(name='Yu Lin'),
            Author(name='Philipp Bucher'),
            Author(name='Bernard M. E. Moret'),
        ],
        doi='arXiv:1307.7919',
        year=2013)

    return DocumentSet([a, b, c, d])


def search_scopus(query):
    """Search Scopus."""
    documents = []
    try:
        retrieved_paper_ids = ScopusSearch(query, view="STANDARD").get_eids()
    except ScopusQueryError:
        print("Impossible to process query \"{}\".".format(query))
        return None
    if len(retrieved_paper_ids) == 0:
        print("No matching documents for the provided query.")
        return None
    for paper_id in retrieved_paper_ids:
        try:
            paper = AbstractRetrieval(paper_id)
        except ValueError:
            print("Impossible to retrieve data for paper \"{}\".".format(paper_id))
            return None
        doc_id = DocumentID()
        doc_id.parse_scopus(paper)
        authors = []
        if paper.authorgroup is not None:
            for author in paper.authorgroup:
                print(author)
                author_affiliation = Affiliation(name=author.affiliation,
                                                 city=author.city,
                                                 country=author.country)
                authors.append(Author(name=author.indexed_name,
                                      orcid=AuthorRetrieval(author.auid).orcid,
                                      affiliations=[author_affiliation]))
        document = Document(id=doc_id,
                            title=paper.title,
                            keywords=paper.authkeywords,
                            abstract=paper.description,
                            source=paper.publicationName,
                            citation_count=paper.citedby_count,
                            year=int(paper.coverDate.split("-")[0]),
                            authors=authors,
                            internal=paper)
        documents.append(document)
    return DocumentSet(docs=documents)
