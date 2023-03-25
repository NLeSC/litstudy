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
from .nlp import \
        generate_topic_cloud, \
        calculate_embedding, \
        compute_word_distribution, \
        TopicModel, \
        Corpus
from .types import DocumentSet
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd


def plot_histogram(
                   data: pd.DataFrame,
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
    """
    This is the general function to plot a histogram (bar plot). All other
    `plot_*_histogram` functions in this module will call this function.

    This function takes a pandas `DataFrame`. Each column represents one
    group and a sequence of bars that will be plotted. The names in the index
    are placed as labels on the axis.

    For instance, a possible input could be
    data frame where the columns are different authors, rows are different
    years, and values are the number of publications per year per author.

    :param ax: The `matplotlib` `Axes` instance the plot will be drawn on. If
               `None`, the current `Axes` instance is used (`plt.gca()`).
    :param vertical: Default bars are horizontal (left to right). Set
                     `vertical=True` for vertical bars (bottom to top).
    :param bar_width: Width of bars. Should be at most `1.0` for 100%.
    :param label_rotation: Rotates the xlabels. This is useful if
                           `vertical=True` since it can be used to place the
                           labels horziontal (`label_rotation=0`), vertical
                           (`label_rotation=90`), or diagonal
                           (`label_rotation=45`).
    :param max_label_length: Labels longer than this length are shortened.
    :param stacked: By default, different groups are drawn next to each other.
                    If `True`, the different groups are stacked on top of each
                    other instead.
    :param legend: Show legend.
    :param relative_to: If not `None`, all bars will be plotted as a
                        percentage relative to this value.
    :param title: Title of plot (set using `ax.set_title`).
    :param xlabel: Title on the X axis (or Y axis if `vertical=True`).
    :param ylabel: Title on the Y axis (or X axis if `vertical=True`).
    """
    if ax is None:
        ax = plt.gca()

    if label_rotation is None:
        if vertical:
            label_rotation = 90
        else:
            label_rotation = 0

    if not ylabel:
        if relative_to is not None:
            ylabel = '% of documents'
        else:
            ylabel = 'No. of documents'

    if not vertical:
        ''' semmyk-research [github]
        ## All plots based on plot_histogram() throws
        ## ValueError: keyword grid_b is not recognized
        ## This is due to changes in matplotlib
        ## reproducibility: python==3.8.16, matplotlib==3.7.1
        ## AKA: assist https://stackoverflow.com/a/75581916/20107918
        
        ### https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.5.0.html#the-first-parameter-of-axes-grid-and-axis-grid-has-been-renamed-to-visible
        ### API change in 3.5.0
        ### The first parameter of Axes.grid and Axis.grid has been renamed to visible
        ### The parameter was previously named b. This deprecation only matters if that parameter was passed using a keyword argument, e.g. grid(b=False)
        '''
        
        #ax.grid(b=False, which='both', axis='y')
        ax.grid(visible=False, which='both', axis='y')
        xlabel, ylabel = ylabel, xlabel
    else:
        #ax.grid(b=False, which='both', axis='x')
        ax.grid(visible=False, which='both', axis='x')

    if title:
        ax.set_title(title)

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

    def fun(docs, **kwargs):
        return compute_groups_histogram(docs, **kwargs).T

    default = dict(title='Categories')
    return wrapper(docs, fun, default, **kwargs)


def plot_year_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents published in each year.

    :param kwargs: Passed to `plot_histogram`.
    """
    default = dict(title='Year of publications')
    return wrapper(docs, compute_year_histogram, default, **kwargs)


def plot_author_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents published per author.

    :param kwargs: Passed to `plot_histogram`.
    """
    default = dict(title='Authors', limit=25)
    return wrapper(docs, compute_author_histogram, default, **kwargs)


def plot_number_authors_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of authors per document.

    :param kwargs: Passed to `plot_histogram`.
    """
    default = dict(title='No. of authors')
    return wrapper(docs, compute_number_authors_histogram, default, **kwargs)


def plot_author_affiliation_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents published per author
    affiliation.

    :param kwargs: Passed to `plot_histogram`.
    """
    default = dict(title='Author + Affiliation', limit=25)
    return wrapper(docs, compute_author_affiliation_histogram, default,
                   **kwargs)


def plot_language_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents by language. """
    default = dict(title='Language')
    return wrapper(docs, compute_language_histogram, default, **kwargs)


def plot_source_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents by publication source. """
    default = dict(title='Publication source', limit=25)
    return wrapper(docs, compute_source_histogram, default, **kwargs)


def plot_source_type_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents by publication source
    type.

    :param kwargs: Passed to `plot_histogram`.
    """
    default = dict(title='Publication source type', limit=25)
    return wrapper(docs, compute_source_type_histogram, default, **kwargs)


def plot_affiliation_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents by author affiliation.

    :param kwargs: Passed to `plot_histogram`.
    """
    default = dict(title='Affiliations', limit=25)
    return wrapper(docs, compute_affiliation_histogram, default, **kwargs)


def plot_country_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents by country of author
    affiliation

    :param kwargs: Passed to `plot_histogram`.
    """
    default = dict(title='Countries', limit=25)
    return wrapper(docs, compute_country_histogram, default, **kwargs)


def plot_continent_histogram(docs: DocumentSet, **kwargs):
    """ Plot histogram of the number of documents by continent of author
    affiliation.

    :param kwargs: Passed to `plot_histogram`.
    """
    default = dict(title='Continents', limit=25)
    return wrapper(docs, compute_continent_histogram, default, **kwargs)


def plot_word_distribution(corpus: Corpus, *, limit=25, **kwargs):
    """ Plot the frequency of the top words in the given corpus.

    :param kwargs: Passed to `plot_histogram`.
    """
    n = len(corpus.frequencies)
    data = compute_word_distribution(corpus, limit=limit)
    return plot_histogram(data, relative_to=n, **kwargs)


def plot_embedding(corpus: Corpus, model: TopicModel, layout=None, ax=None):
    """ TODO """
    if ax is None:
        ax = plt.gca()

    if layout is None:
        layout = calculate_embedding(corpus)

    num_topics = len(model.topic2token)
    best_topic = np.argmax(model.doc2topic.T, axis=0)

    colors = seaborn.color_palette('hls', num_topics)
    colors = np.array(colors)[:, :3] * 0.9  # Mute colors a bit

    for i in range(num_topics):
        indices = best_topic == i
        # label = 'ABCDEFGHIJLMNOPQRSTUVWXYZ'[i]
        label = i + 1

        for j in np.argwhere(indices)[:, 0]:
            x, y = layout[j]
            ax.scatter(
                    x,
                    y,
                    marker='o',
                    s=150,
                    linewidth=0.5,
                    color=colors[i],
                    zorder=2*j,
            )

            ax.text(
                x,
                y,
                label,
                fontsize=6,
                color='1',
                va='center',
                ha='center',
                fontweight='bold',
                zorder=2*j + 1,
            )

        top_items = model.best_tokens_for_topic(i, limit=3)
        label = f'Topic {label}:' + ', '.join(top_items)

        center = np.median(layout[indices], axis=0)
        ax.text(
                center[0],
                center[1],
                label,
                va='center',
                ha='center',
                color='1',
                backgroundcolor=(0, 0, 0, .75),
                zorder=10 * len(best_topic),
        )

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_topic_clouds(model: TopicModel, *, fig=None, ncols=3, **kwargs):
    """ Plot word clouds for each of the topics from the given topic model. """
    if fig is None:
        plt.clf()
        fig = plt.gcf()

    nrows = math.ceil(model.num_topics / float(ncols))

    for i in range(model.num_topics):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot_topic_cloud(model, i, ax=ax, **kwargs)


def plot_topic_cloud(model: TopicModel, topic_id: int, *, ax=None, **kwargs):
    """ Plot a word cloud for the given topic from the given topic model. """
    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])

    im = generate_topic_cloud(model, topic_id, **kwargs).to_array()
    ax.set_title(f'Topic {topic_id + 1}')
    ax.imshow(im, interpolation='bilinear')


def plot_document_topics(model: TopicModel, document_id: int, *, ax=None):
    if ax is None:
        ax = plt.gca()

    weights = model.document_topics(document_id)

    ax.bar(
            np.arange(model.num_topics),
            model.document_topics(document_id)
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel('Relevance')
    ax.set_ylabel('Topics')

    for i, w in enumerate(weights):
        if w > 0.01:
            plt.text(i, w, str(i + 1), ha='center', va='bottom')
