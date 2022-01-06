from .common import FuzzyMatcher
from collections import defaultdict, OrderedDict
import pandas as pd


def histogram(docs, fun, keys=None, sort_by_key=False, groups=None,
              limit=None):
    if isinstance(groups, list):
        data = dict((v, docs.data.eval(v)) for v in groups)
        groups = pd.DataFrame(data)
    elif isinstance(groups, dict):
        data = dict((name, docs.data.eval(expr))
                    for name, expr in groups.items())
        groups = pd.DataFrame(data)
    elif groups is None:
        groups = pd.DataFrame(index=range(len(docs)))
    elif isinstance(groups, pd.DataFrame):
        pass

    assert len(groups) == len(docs)
    totals = defaultdict(lambda: 0)
    counts = defaultdict(lambda: 0)

    for doc, row in zip(docs, groups.itertuples()):
        items = fun(doc)

        if items is None:
            continue

        for item in items:
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


def compute_year_histogram(docs, **kwargs):
    """ Returns data frame """
    years = [doc.publication_year for doc in docs]
    min_year = min(year for year in years if year)
    max_year = max(year for year in years if year)
    keys = list(range(min_year, max_year + 1))

    def extract(doc):
        y = doc.publication_year
        return [y] if y else []

    return histogram(docs, extract, keys=keys, **kwargs)


def compute_number_authors_histogram(docs, max_authors=10, **kwargs):
    keys = ['NA'] + list(range(1, max_authors + 1)) + [f'>{max_authors}']

    def extract(doc):
        n = len(doc.authors or [])
        if n == 0:
            return ['NA']
        elif n > max_authors:
            return [f'>{max_authors}']
        else:
            return [n]

    return histogram(docs, extract, keys=keys, **kwargs)


def compute_author_histogram(docs, **kwargs):
    def extract(doc):
        return [a.name for a in doc.authors or []]

    return histogram(docs, extract, **kwargs)


def compute_author_affiliation_histogram(docs, **kwargs):
    def extract(doc):
        result = []
        for author in doc.authors or []:
            for affiliation in author.affiliations or []:
                if author.name and affiliation.name:
                    result.append(f'{author.name}, {affiliation.name}')
        return result

    return histogram(docs, extract, **kwargs)


def compute_language_histogram(docs, **kwargs):
    def extract(doc):
        return [doc.language] if doc.language else []

    return histogram(docs, extract, **kwargs)


def default_mapper(mapper):
    if mapper is None:
        return FuzzyMatcher()
    elif type(mapper) is dict:
        return FuzzyMatcher(mapper)
    else:
        return mapper


def compute_source_histogram(docs, mapper=None, **kwargs):
    mapper = default_mapper(mapper)

    def extract(doc):
        source = doc.publication_source
        return [mapper.get(source) if source else '(unknown)']

    return histogram(docs, extract, **kwargs)


def compute_affiliation_histogram(docs, mapper=None, **kwargs):
    mapper = default_mapper(mapper)

    def extract(doc):
        result = set()
        for author in doc.authors or []:
            for aff in author.affiliations or []:
                if aff.name:
                    result.add(aff.name)

        return result

    return histogram(docs, extract, **kwargs)


def compute_country_histogram(docs, **kwargs):
    def extract(doc):
        result = set()
        for author in doc.authors or []:
            for aff in author.affiliations or []:
                if aff.country:
                    result.add(aff.country)

        return result

    return histogram(docs, extract, **kwargs)
