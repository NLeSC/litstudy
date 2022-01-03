from collections import defaultdict
from .common import FuzzyMatcher
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_bars(keys, values, *, title='', xlabel='', ylabel=None,
              relative_to=None, ax=None, vertical=False):
    assert len(keys) == len(values)

    if ax is None:
        ax = plt.gca()

    if title:
        ax.set_title(title)

    if ylabel is None:
        if relative_to is None:
            ylabel = 'No. of documents'
        else:
            ylabel = '% of documents'

    if relative_to is not None:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        values = [v / float(relative_to) * 100 for v in values]

    if not vertical:
        if xlabel:
            ax.set_ylabel(xlabel)

        if ylabel:
            ax.set_xlabel(ylabel)
    else:
        if xlabel:
            ax.set_xlabel(xlabel)

        if ylabel:
            ax.set_ylabel(ylabel)

    def shorten(label):
        return label[:100] + '...' if len(label) > 100 else label

    n = len(keys)
    labels = [shorten(str(key)) for key in keys]

    if not vertical:
        values = values[::-1]
        labels = labels[::-1]

        ax.grid(b=False, which='both', axis='y')
        ax.barh(range(n), values, tick_label=labels)
    else:
        ax.grid(b=False, which='both', axis='x')
        ax.bar(range(n), values, tick_label=labels)


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
