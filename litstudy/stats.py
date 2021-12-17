from collections import defaultdict, OrderedDict
from .clean import FuzzyMatcher
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_bars(data, *, title='', xlabel='', ylabel=None, ax=None, vertical=False):
    if ax is None:
        ax = plt.gca()

    data = pd.DataFrame(data)
    #sns.barplot(data=data, hue='')
    data.plot.bar(ax=ax)


def histogram(docs, fun, sort_by_key=False, groups=None, limit=None):
    if groups is None:
        groups = pd.DataFrame(index=range(len(docs)))
    else:
        groups = pd.DataFrame(groups)

    assert len(groups) == len(docs)
    totals = defaultdict(lambda: 0)
    counts = defaultdict(lambda: np.zeros(len(groups.columns), dtype=np.int64))

    for doc, row in zip(docs, groups.itertuples(index=False)):
        items = fun(doc)

        if items is None:
            continue

        for item in items:
            if item is None:
                continue

            totals[item] += 1

            for index, weight in enumerate(row):
                counts[item][index] += weight

    totals = pd.Series(totals)
    totals = totals.sort_values(ascending=False)

    if limit is not None:
        totals = totals[:limit]

    if sort_by_key:
        totals = totals.sort_index()

    data = [counts[key] for key in totals.index]
    columns = groups.columns

    if not len(columns):
        data = [totals]
        columns = ['Frequency']

    return pd.DataFrame(
            data,
            index=totals.index,
            columns=columns,
    )





def plot_year_histogram(docs, **kwargs):
    count = defaultdict(int)

    for doc in docs:
        year = doc.publication_year

        if year is not None:
            count[year] += 1

    # No data
    if not count:
        return

    min_year = min(count.keys())
    max_year = max(count.keys())
    years = list(range(min_year, max_year + 1))

    return plot_bars(years, [count[y] for y in years], **kwargs)


def plot_number_authors_histogram(docs, max_authors=10, **kwargs):
    count = defaultdict(int)
    unknown = 0
    overflow = 0

    # No data
    if not docs:
        return

    for doc in docs:
        if not doc.authors:
            unknown += 1
        elif len(doc.authors) > max_authors:
            overflow += 1
        else:
            count[len(doc.authors)] += 1

    numbers = list(range(1, max_authors))
    keys = ['NA'] + numbers + [f'>{max_authors}']
    values = [unknown] + [count[n] for n in numbers] + [overflow]
    return plot_bars(keys, values, **kwargs)


def plot_histogram(docs, fun, top_k=25, percentage=False, **kwargs):
    count = defaultdict(int)

    for doc in docs:
        output = fun(doc)

        if output is None:
            continue

        if type(output) is str:
            output = [output]

        for item in output:
            if item is not None:
                key = str(item).strip()
                count[key] += 1

    items = list(count.items())
    items.sort(key=lambda p: p[1], reverse=True)

    if top_k is not None and len(items) > top_k:
        items = items[:top_k]

    keys = [k for k, v in items]
    values = [v for k, v in items]

    if percentage:
        total = sum(count.values())
    else:
        total = None

    return plot_bars(keys, values, relative_to=total, **kwargs)


def plot_author_histogram(docs, **kwargs):
    def extract(doc):
        return [a.name for a in doc.authors or []]

    return plot_histogram(docs, extract, **kwargs)


def plot_author_affiliation_histogram(docs, **kwargs):
    def extract(doc):
        result = []
        for author in doc.authors or []:
            for affiliation in author.affiliations or []:
                if author.name and affiliation.name:
                    result.append(f'{author.name}, {affiliation.name}')
        return result

    return plot_histogram(docs, extract, **kwargs)


def plot_language_histogram(docs, **kwargs):
    def extract(doc):
        return [doc.language] if doc.language else []

    return plot_histogram(docs, extract, **kwargs)


def default_mapper(mapper):
    if mapper is None:
        return FuzzyMatcher()
    elif type(mapper) is dict:
        return FuzzyMatcher(mapper)
    else:
        return mapper

def plot_source_histogram(docs, mapper=None, **kwargs):
    mapper = default_mapper(mapper)

    def extract(doc):
        source = doc.publication_source

        if source is not None:
            return mapper.get(source)
        else:
            return '(unknown)'

    return plot_histogram(docs, extract, **kwargs)


def plot_affiliation_histogram(docs, mapper=None, **kwargs):
    mapper = default_mapper(mapper)

    def extract(doc):
        result = set()
        for author in doc.authors or []:
            for affiliation in author.affiliations or []:
                name = affiliation.name

                if name is not None:
                    name = mapper.get(name)
                else:
                    name = '(unknown)'

                result.add(name)

        return result

    return plot_histogram(docs, extract, **kwargs)

def plot_country_histogram(docs, **kwargs):
    def extract(doc):
        result = set()
        for author in doc.authors or []:
            for affiliation in author.affiliations or []:
                result.add(affiliation.country)

        return result

    return plot_histogram(docs, extract, **kwargs)
