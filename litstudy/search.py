from pybliometrics.scopus import ScopusSearch, AbstractRetrieval, AuthorRetrieval, ContentAffiliationRetrieval
from pybliometrics.scopus.exception import ScopusQueryError
from tqdm import tqdm
import requests
from urllib.parse import quote_plus
import bibtexparser
import iso639

from .common import Document, DocumentID, DocumentSet, Author, Affiliation


def search_mockup():
    a = Document(
            title=' A unified analytical theory of heteropolymers for sequence-specific'
                  ' phase behaviors of polyelectrolytes and polyampholytes ',
            authors=[
                Author(name='Yi-Hsuan Lin',
                       affiliations=[Affiliation(name='University of affiliation1'),
                                     Affiliation(name='affiliation3')]),
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

        if paper.language:
            language = iso639.languages.get(part2b=paper.language).name
        else:
            language = None

        document = Document(id=doc_id,
                            title=paper.title,
                            keywords=paper.authkeywords,
                            abstract=paper.description,
                            source=paper.publicationName,
                            source_type=paper.aggregationType,
                            citation_count=int(paper.citedby_count),
                            language=language,
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
    query = quote_plus(query)
    request = requests.get("http://dblp.org/search/publ/api?format=json&h=1000&f=0&q={}".format(query))
    results = request.json()
    expected_documents = int(results["result"]["hits"]["@total"])
    for paper in results["result"]["hits"]["hit"]:
        retrieved_papers.append(paper)
    while len(retrieved_papers) < expected_documents:
        if int(results["result"]["hits"]["@total"]) > int(results["result"]["hits"]["@sent"]):
            request = requests.get("http://dblp.org/search/publ/api?format=json&h=1000&f={}&q={}"
                                   .format(len(retrieved_papers), query.replace(" ", "+")))
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
            document.year = int(paper["info"]["year"])
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
        authors = []
        try:
            if type(paper["info"]["authors"]["author"]) is str:
                authors.append(Author(name=paper["info"]["authors"]["author"]))
            else:
                for author in paper["info"]["authors"]["author"]:
                    authors.append(Author(name=author))
        except KeyError:
            pass
        document.authors = authors
        documents.append(document)
    if docs:
        return DocumentSet(docs=documents).union(docs)
    else:
        return DocumentSet(docs=documents)


def load_bibtex(file, lookup_authors=False):
    """Load the content of a BibTex file."""

    documents = []
    with open(file) as bibtex_file:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        bibtex_data = bibtexparser.load(bibtex_file, parser=parser)
    bibtex_file.close()
    for paper in tqdm(bibtex_data.entries):
        doc_id = DocumentID()
        doc_id.parse_bibtex(paper)
        document = Document(id=doc_id,
                            title=paper["title"],
                            internal=paper)
        try:
            document.abstract = paper["abstract"]
        except KeyError:
            pass
        try:
            document.year = int(paper["year"])
        except KeyError:
            pass
        try:
            document.source = paper["journal"]
        except KeyError:
            pass
        try:
            document.source_type = paper["ENTRYTYPE"]
        except KeyError:
            pass
        try:
            document.publisher = paper["publisher"]
        except KeyError:
            pass
        try:
            document.keywords = paper["keywords"]
        except KeyError:
            pass
        authors = []
        if lookup_authors:
            if document.id.is_doi:
                request = requests.get("http://api.semanticscholar.org/v1/paper/{}".format(quote_plus(document.id.id)))
                results = request.json()
                try:
                    for author in results["authors"]:
                        authors.append(Author(name=author["name"]))
                except KeyError:
                    pass
        else:
            try:
                for author in paper["author"].split("and"):
                    authors.append(Author(name=author.strip("{}")))
            except KeyError:
                pass
        document.authors = authors
        documents.append(document)
    return documents


def query_semanticscholar(documents):
    for document in tqdm(documents):
        if document.id.is_doi:
            request = requests.get("http://api.semanticscholar.org/v1/paper/{}".format(quote_plus(document.id.id)))
            results = request.json()
            if not document.title:
                try:
                    document.title = results["title"]
                except KeyError:
                    pass
            if len(document.authors) == 0:
                try:
                    for author in results["authors"]:
                        document.authors.append(Author(name=author["name"]))
                except KeyError:
                    pass
            if not document.abstract:
                try:
                    document.abstract = results["abstract"]
                except KeyError:
                    pass
            if not document.references or len(document.references) == 0:
                try:
                    references = []
                    for reference in results["references"]:
                        references.append(reference["title"])
                    document.references = references
                except KeyError:
                    pass
            if not document.year:
                try:
                    document.year = int(results["year"])
                except KeyError:
                    pass
            if not document.source:
                try:
                    document.source = results["venue"]
                except KeyError:
                    pass
            if not document.citation_count:
                try:
                    document.citation_count = len(results["citations"])
                except KeyError:
                    pass


def query_crossref(documents):
    for document in tqdm(documents):
        if document.id.is_doi:
            request = requests.get("https://api.crossref.org/v1/works/{}".format(quote_plus(document.id.id)))
            if request.status_code != 200:
                continue
            results = request.json()

            if not document.title:
                try:
                    document.title = results["message"]["title"][0]
                except KeyError:
                    pass
            if len(document.authors) == 0:
                try:
                    for author in results["message"]["author"]:
                        document.authors.append(Author(name=(author["message"]["given"] + author["message"]["family"])))
                except KeyError:
                    pass
            if not document.year:
                try:
                    document.year = int(results["message"]["published-print"]["date-parts"][0])
                except KeyError:
                    pass
            if not document.source:
                try:
                    document.source = results["message"]["container-title"][0]
                except KeyError:
                    pass
            if not document.source_type:
                try:
                    document.source_type = results["message"]["published-print"]["type"]
                except KeyError:
                    pass
            if not document.citation_count:
                try:
                    document.citation_count = int(results["message"]["is-referenced-by-count"])
                except KeyError:
                    pass
            if not document.language:
                try:
                    document.language = results["message"]["language"]
                except KeyError:
                    pass
            if not document.publisher:
                try:
                    document.publisher = results["message"]["publisher"]
                except KeyError:
                    pass
    return DocumentSet(documents)
