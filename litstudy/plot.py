from .stats import \
        compute_year_histogram, \
        compute_author_histogram, \
        compute_author_affiliation_histogram, \
        compute_language_histogram, \
        compute_number_authors_histogram, \
        compute_source_histogram, \
        compute_source_type_histogram, \
        compute_affiliation_histogram, \
        compute_country_histogram, \
        compute_continent_histogram, \
        compute_groups_histogram
import matplotlib.pyplot as plt
import numpy as np
import inspect


def plot_histogram(
                   data,
                   keys=None,
                   title='',
                   xlabel='',
                   ylabel='',
                   label_rotation=None,
                   vertical=False,
                   bar_width=0.8,
                   max_label_length=100,
                   stacked=False,
                   legend=True,
                   relative_to=None,
                   ax=None):
    if ax is None:
        ax = plt.gca()

    if label_rotation is None:
        if vertical:
            label_rotation = 90
        else:
            label_rotation = 0

    if title:
        ax.set_title(title)

    if not ylabel:
        if relative_to is not None:
            ylabel = '% of documents'
        else:
            ylabel = 'No. of documents'

    if not vertical:
        ax.grid(b=False, which='both', axis='y')

        if xlabel:
            ax.set_ylabel(xlabel)

        if ylabel:
            ax.set_xlabel(ylabel)
    else:
        ax.grid(b=False, which='both', axis='x')

        if xlabel:
            ax.set_xlabel(xlabel)

        if ylabel:
            ax.set_ylabel(ylabel)

    if keys is None:
        keys = list(data.index)

    n = len(keys)
    groups = list(data)

    if not groups or not keys:
        return

    bar_bottom = np.array([0] * len(keys))

    for i, group in enumerate(groups):
        ys = data[group]
        bottom = bar_bottom

        if relative_to is not None:
            ys = ys / relative_to * 100

        if stacked:
            bar_bottom = bar_bottom + ys
            xs = np.arange(n) - 0.5 * bar_width
            bin_width = bar_width
        else:
            bin_width = bar_width / len(groups)
            xs = np.arange(n) + i * bin_width - 0.5 * bar_width

        if not vertical:
            ax.barh(xs, ys, label=group, height=bin_width, left=bottom,
                    align='edge')
        else:
            ax.bar(xs, ys, label=group, width=bin_width, bottom=bottom,
                   align='edge')

    if legend and len(groups) > 1:
        ax.legend()

    def shorten(label):
        label = str(label)
        n = max_label_length
        return label[:n] + '...' if len(label) > n else label

    labels = list(map(shorten, keys))

    if not vertical:
        ax.set_ylim((n - 0.5, -0.5))
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, rotation=label_rotation)
    else:
        ax.set_xlim((-0.5, n - 0.5))
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=label_rotation, ha='right')

    return ax


def wrapper(docs, fun, default, **kwargs):
    for key, value in default.items():
        kwargs.setdefault(key, value)

    plot_kwargs = dict()
    params = inspect.signature(plot_histogram).parameters

    for key in list(kwargs):
        if key in params:
            plot_kwargs[key] = kwargs.pop(key)

    if kwargs.pop('relative', False):
        plot_kwargs['relative_to'] = len(docs)

    data = fun(docs, **kwargs)
    return plot_histogram(data, **plot_kwargs)


def plot_groups_histogram(docs, **kwargs):
    """ """
    default = dict(title='Categories')
    return wrapper(docs, lambda docs, **kwargs: compute_groups_histogram(docs, **kwargs).T, default, **kwargs)


def plot_year_histogram(docs, **kwargs):
    """ """
    default = dict(title='Year of publications')
    return wrapper(docs, compute_year_histogram, default, **kwargs)


def plot_author_histogram(docs, **kwargs):
    """ """
    default = dict(title='Authors', limit=25)
    return wrapper(docs, compute_author_histogram, default, **kwargs)


def plot_number_authors_histogram(docs, **kwargs):
    """ """
    default = dict(title='No. of authors')
    return wrapper(docs, compute_number_authors_histogram, default, **kwargs)


def plot_author_affiliation_histogram(docs, **kwargs):
    """ """
    default = dict(title='Author + Affiliation', limit=25)
    return wrapper(docs, compute_author_affiliation_histogram, default,
                   **kwargs)


def plot_language_histogram(docs, **kwargs):
    """ """
    default = dict(title='Language')
    return wrapper(docs, compute_language_histogram, default, **kwargs)


def plot_source_histogram(docs, **kwargs):
    """ """
    default = dict(title='Publication source', limit=25)
    return wrapper(docs, compute_source_histogram, default, **kwargs)


def plot_source_type_histogram(docs, **kwargs):
    """ """
    default = dict(title='Publication source type', limit=25)
    return wrapper(docs, compute_source_type_histogram, default, **kwargs)


def plot_affiliation_histogram(docs, **kwargs):
    """ """
    default = dict(title='Affiliations', limit=25)
    return wrapper(docs, compute_affiliation_histogram, default, **kwargs)


def plot_country_histogram(docs, **kwargs):
    """ """
    default = dict(title='Countries', limit=25)
    return wrapper(docs, compute_country_histogram, default, **kwargs)


def plot_continent_histogram(docs, **kwargs):
    """ """
    default = dict(title='Continents', limit=25)
    return wrapper(docs, compute_continent_histogram, default, **kwargs)
