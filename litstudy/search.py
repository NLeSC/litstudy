from pybliometrics.scopus import ScopusSearch, AbstractRetrieval, AuthorRetrieval, ContentAffiliationRetrieval
from pybliometrics.scopus.exception import ScopusQueryError
from tqdm import tqdm
import requests

from .common import Document, DocumentID, DocumentSet, Author, Affiliation


def search_mockup():
    a = Document(
            title=' A unified analytical theory of heteropolymers for sequence-specific phase behaviors of polyelectrolytes and polyampholytes ',
            authors=[
                Author(name='Yi-Hsuan Lin', affiliations=[Affiliation(name='University of affiliation1'), Affiliation(name='affiliation3')]),
                Author(name='Jacob P. Brady', affiliations=[Affiliation(name='University of affiliation1')]),
                Author(name='Hue Sun Chan', affiliations=[Affiliation(name='affiliation2')]),
                Author(name='Kingshuk Ghosh'),
            ],
            id='arXiv:1908.09726',
            year=2019,
            source_type='type1',
            source='International Conference on 1')

    b = Document(
            title='Scaling methods for accelerating kinetic Monte Carlo simulations of chemical reaction networks',
            authors=[
                Author(name='Yen Ting Lin', affiliations=[Affiliation(name='affiliation2')]),
                Author(name='Song Feng', affiliations=[Affiliation(name='affiliation3')]),
                Author(name='William S. Hlavacek'),
            ],
            id='1',
            year=2019,
            source_type='type2',
            source='International Conference on 2')

    c = Document(
            title='Discreteness of populations enervates biodiversity in evolution',
            authors=[
                Author(name='Yen-Chih Lin'),
                Author(name='Tzay-Ming Hong'),
                Author(name='Hsiu-Hau Lin'),
            ],
            id='arXiv:1005.4335',
            year=2010,
            source_type='type2',
            source='International Conference on 2')

    d = Document(
            title='Phylogenetic Analysis of Cell Types using Histone Modifications',
            authors=[
                Author(name='Nishanth Ulhas Nair'),
                Author(name='Yu Lin'),
                Author(name='Philipp Bucher'),
                Author(name='Bernard M. E. Moret'),
            ],
            id='arXiv:1307.7919',
            year=2013,
            source_type='type3',
            source='International Conference on 3')
    
    return DocumentSet([a, b, c, d])


def search_scopus(query, docs=None):
    """Search Scopus."""

    documents = []
    authors_cache = {}
    affiliations_cache = {}
    try:
        retrieved_paper_ids = ScopusSearch(query, view="STANDARD").get_eids()
    except ScopusQueryError:
        print("Impossible to process query \"{}\".".format(query))
        return None
    if len(retrieved_paper_ids) == 0:
        print("No matching documents for the provided query.")
        return None
    for paper_id in tqdm(retrieved_paper_ids):
        try:
            paper = AbstractRetrieval(paper_id, view="FULL")
        except ValueError:
            print("Impossible to retrieve data for paper \"{}\".".format(paper_id))
            return None
        doc_id = DocumentID()
        doc_id.parse_scopus(paper)
        authors = []
        if paper.authors:
            for author in paper.authors:
                author_affiliations = []
                if author.auid in authors_cache:
                    authors.append(Author(name=author.indexed_name,
                                          orcid=authors_cache[author.auid],
                                          affiliations=author_affiliations))
                else:
                    authors_cache[author.auid] = AuthorRetrieval(author.auid).orcid
                    authors.append(Author(name=author.indexed_name,
                                          orcid=authors_cache[author.auid],
                                          affiliations=author_affiliations))
                if author.affiliation:
                    for affiliation_id in author.affiliation:
                        if affiliation_id in affiliations_cache:
                            affiliation = affiliations_cache[affiliation_id]
                        else:
                            affiliation = ContentAffiliationRetrieval(affiliation_id)
                            affiliations_cache[affiliation_id] = affiliation
                        author_affiliations.append(Affiliation(name=affiliation.affiliation_name,
                                                               city=affiliation.city,
                                                               country=affiliation.country))
        references = []
        if paper.refcount and int(paper.refcount) > 0:
            for reference in paper.references:
                if reference.title:
                    references.append(reference.title)
        document = Document(id=doc_id,
                            title=paper.title,
                            keywords=paper.authkeywords,
                            abstract=paper.description,
                            source=paper.publicationName,
                            source_type=paper.aggregationType,
                            citation_count=paper.citedby_count,
                            language=paper.language,
                            year=int(paper.coverDate.split("-")[0]),
                            authors=authors,
                            references=references,
                            publisher=paper.publisher,
                            internal=paper)
        documents.append(document)
    if docs:
        return DocumentSet(docs=documents).union(docs)
    else:
        return DocumentSet(docs=documents)


def search_dblp(query, docs=None):
    """Search DBLP."""

    documents = []
    retrieved_papers = []
    request = requests.get("http://dblp.org/search/publ/api?format=json&h=1000&f=0&q={}".format(query.replace(" ", "+")))
    results = request.json()
    expected_documents = int(results["result"]["hits"]["@total"])
    for paper in results["result"]["hits"]["hit"]:
        retrieved_papers.append(paper)
    while len(retrieved_papers) < expected_documents:
        if int(results["result"]["hits"]["@total"]) > int(results["result"]["hits"]["@sent"]):
            request = requests.get("http://dblp.org/search/publ/api?format=json&h=1000&f={}&q={}".format(len(retrieved_papers), query.replace(" ", "+")))
            results = request.json()
            for paper in results["result"]["hits"]["hit"]:
                retrieved_papers.append(paper)
    for paper in tqdm(retrieved_papers):
        doc_id = DocumentID()
        doc_id.parse_dblp(paper)
        document = Document(id=doc_id,
                            title=paper["info"]["title"],
                            internal=paper)
        try:
            document.year = paper["info"]["year"]
        except KeyError:
            pass
        try:
            document.source = paper["info"]["venue"]
        except KeyError:
            pass
        try:
            document.source_type = paper["info"]["type"]
        except KeyError:
            pass
        try:
            document.publisher = paper["info"]["publisher"]
        except KeyError:
            pass
        try:
            authors = []
            if type(paper["info"]["authors"]["author"]) is str:
                authors.append(Author(name=paper["info"]["authors"]["author"]))
            else:
                for author in paper["info"]["authors"]["author"]:
                    authors.append(Author(name=author))
            document.authors = authors
        except KeyError:
            pass
        documents.append(document)
    if docs:
        return DocumentSet(docs=documents).union(docs)
    else:
        return DocumentSet(docs=documents)
