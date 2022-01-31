from .common import FuzzyMatcher
from collections import defaultdict, OrderedDict
from .types import DocumentSet
import pandas as pd


def compute_histogram(docs, fun, keys=None, sort_by_key=False, groups=None,
                      limit=None):
    if isinstance(groups, list):
        groups = dict((v, v) for v in groups)

    if isinstance(groups, dict):
        data = dict()

        for name, value in groups.items():
            if isinstance(value, str):
                data[name] = docs.data.eval(value)
            else:
                values = list(value)
                assert len(values) == len(docs)
                data[name] = values

        groups = pd.DataFrame(data)
    elif groups is None:
        groups = pd.DataFrame(index=range(len(docs)))
    elif isinstance(groups, pd.DataFrame):
        pass

    assert len(groups) == len(docs)
    totals = defaultdict(lambda: 0)
    counts = defaultdict(lambda: 0)

    for doc, row in zip(docs, groups.itertuples()):
        for item in fun(doc):
            if item is None:
                continue

            totals[item] += 1

            for index, weight in enumerate(row[1:]):
                if weight:
                    counts[item, index] += weight

    if keys is None:
        totals = pd.Series(totals)
        totals = totals.sort_values(ascending=False)

        if limit is not None:
            totals = totals[:limit]

        if sort_by_key:
            totals = totals.sort_index()
    else:
        totals = OrderedDict((k, totals[k]) for k in keys)
        totals = pd.Series(totals)

    columns = groups.columns
    if len(columns):
        data = [[counts[item, index]
                for index in range(len(columns))]
                for item in totals.index]
    else:
        columns = ['Frequency']
        data = totals

    return pd.DataFrame(
            data,
            index=totals.index,
            columns=columns,
    )


def compute_groups_histogram(docs: DocumentSet, **kwargs) -> pd.DataFrame:
    return compute_histogram(docs, lambda _: ['Frequency'], **kwargs).T


def compute_year_histogram(docs: DocumentSet, **kwargs) -> pd.DataFrame:
    """ Compute a histogram of the number of documents published in each
    year. """
    years = [doc.publication_year for doc in docs]
    min_year = min(year for year in years if year)
    max_year = max(year for year in years if year)
    keys = list(range(min_year, max_year + 1))

    def extract(doc):
        yield doc.publication_year

    return compute_histogram(docs, extract, keys=keys, **kwargs)


def compute_number_authors_histogram(docs: DocumentSet, max_authors=10,
                                     **kwargs) -> pd.DataFrame:
    """ Compute a histogram of the number of authors per document.

    :param max_authors: If a document has more than `max_author` authors, it
                        is it is listed as a special "max authors" category.
    """
    keys = ['NA'] + list(range(1, max_authors + 1)) + [f'>{max_authors}']

    def extract(doc):
        n = len(doc.authors or [])
        if n == 0:
            yield 'NA'
        elif n > max_authors:
            yield '>{max_authors}'
        else:
            yield n

    return compute_histogram(docs, extract, keys=keys, **kwargs)


def compute_language_histogram(docs: DocumentSet, **kwargs) -> pd.DataFrame:
    """ Compute a histogram of number of documents by language. """
    def extract(doc):
        if doc.language:
            yield doc.language

    return compute_histogram(docs, extract, **kwargs)


def default_mapper(mapper):
    if mapper is None:
        return FuzzyMatcher()
    elif type(mapper) is dict:
        return FuzzyMatcher(mapper)
    else:
        return mapper


def compute_source_histogram(docs: DocumentSet, mapper=None, **kwargs
                             ) -> pd.DataFrame:
    """ Compute a histogram of number of documents by publication source. """
    mapper = default_mapper(mapper)

    def extract(doc):
        source = doc.publication_source
        yield mapper.get(source) if source else '(unknown)'

    return compute_histogram(docs, extract, **kwargs)


def compute_source_type_histogram(docs: DocumentSet, **kwargs
                                  ) -> pd.DataFrame:
    """ Compute a histogram of number of documents by source type. """
    def extract(doc):
        yield doc.source_type or '(unknown)'

    return compute_histogram(docs, extract, **kwargs)


def compute_author_histogram(docs: DocumentSet, **kwargs) -> pd.DataFrame:
    """ Compute a histogram of number of documents by author name. """
    def extract(doc):
        for author in doc.authors or []:
            yield a.name

    return compute_histogram(docs, extract, **kwargs)


def compute_author_affiliation_histogram(docs: DocumentSet, **kwargs
                                         ) -> pd.DataFrame:
    """ Compute a histogram of number of documents by (author name,
    affiliation name) combinations. This can help reduce conflicts where there
    are many authors of the same name working for different affiliations."""
    def extract(doc):
        result = []
        for author in doc.authors or []:
            for affiliation in author.affiliations or []:
                if author.name and affiliation.name:
                    yield f'{author.name}, {affiliation.name}'

    return compute_histogram(docs, extract, **kwargs)


def compute_affiliation_histogram(docs: DocumentSet, mapper=None, **kwargs
                                  ) -> pd.DataFrame:
    """ Compute a histogram of number of documents by affiliation name. """
    mapper = default_mapper(mapper)

    def extract(doc):
        result = set()
        for author in doc.authors or []:
            for aff in author.affiliations or []:
                if aff.name:
                    result.add(mapper.get(aff.name))

        return result

    return compute_histogram(docs, extract, **kwargs)


def extract_country(aff):
    from .continent import COUNTRY_TO_CONTINENT

    # Sometimes affiliation has given country
    if country := aff.country:
        return country

    # Sometimes the country is in the affiliation name
    name = aff.name
    if name:
        for country in COUNTRY_TO_CONTINENT.keys():
            if country in name:
                return country

    return None


def compute_country_histogram(docs: DocumentSet, **kwargs) -> pd.DataFrame:
    """ Compute a histogram of number of documents by affiliation country. """
    def extract(doc):
        result = set()
        for author in doc.authors or []:
            for aff in author.affiliations or []:
                if country := extract_country(aff):
                    result.add(country)

        return result

    return compute_histogram(docs, extract, **kwargs)


def compute_continent_histogram(docs: DocumentSet, **kwargs) -> pd.DataFrame:
    """ Compute a histogram of number of documents by affiliation
        continent.
    """
    from .continent import COUNTRY_TO_CONTINENT

    def extract(doc):
        result = set()
        for author in doc.authors or []:
            for aff in author.affiliations or []:
                if country := extract_country(aff):
                    country = country.strip().lower()

                    if country.startswith('the '):
                        country = country[4:]

                    if continent := COUNTRY_TO_CONTINENT.get(country):
                        result.add(continent)
                    else:
                        result.add('Other')

        return result

    return compute_histogram(docs, extract, **kwargs)
