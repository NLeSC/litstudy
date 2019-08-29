import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
# import pandas as pd
from sklearn.decomposition import TruncatedSVD
import sklearn.manifold
import numpy as np
import math
import wordcloud

from .nlp import create_tfidf

sns.set('paper')

#-----------------------------------------------------------------------
# Statistics plotting functions
#-----------------------------------------------------------------------

def top_k(mapping, k=10):
    return sorted(mapping.keys(), key=lambda x: mapping[x])[::-1][:k]

def prepare_fig(w=1, h=None, wordcloud=False):
    if h is None: h = w
    fig = plt.figure(figsize=(6 * w, 3 * h))
    ax = plt.gca()
    if wordcloud is True:
        fig.clear()
    return fig, ax

# Publications per aggregation type
def plot_statistic(fun, docset, x=None, ax=None, x_label="", count=None):
    """ Plot statistics of some sort in a bar plot. If x is not given,
        (None) all keys with a count > 0 are plotted. If x is a list, the
        counts of all list elements are included. If x is an integer,
        the x keys with highest counts are plotted. """

    if ax is None:
        fig, ax = prepare_fig(2)

    # Use given count dict if we are plotting something
    # unrelated to documents and the counting has already been performed.
    if count is None:
        count = defaultdict(int)
        for d in docset.docs:
            for key in fun(d):
                if key:
                    count[str(key)] += 1

    ax.set_xlabel(x_label)
    
    if type(x) == type([]):
        keys = x
    elif type(x) == type(1):
        keys = top_k(count, x)
    else:
        keys = list(count.keys())

    ax.barh(keys,
        [count[str(a)] for a in keys],
        tick_label=[str(key)[:50] for key in keys])

    plt.show()

def clean_affiliation(name):
    name = str(name).title()
    pairs = [
        ['University', 'U'],
        ['Universitat', 'U'],
        ['Laboratories', 'Lab'],
        ['Laboratory', 'Lab'],
        ['National', 'Nat'],
        ['Corporation', 'Corp'],
        ['Technology', 'Tech'],
        ['Institute', 'Inst'],
    ]
    
    for needle, replacement in pairs:
        name = name.replace(needle, replacement)
    return name

def affiliation_to_type(name):
    name = name.lower()
    pairs = [
        ['universi', 'Academic institute'],
        ['hochschule', 'Academic institute'],
        ['school', 'Academic institute'],
        ['ecole', 'Academic institute'],
        ['institute', 'Academic institute'],
        ['research center', 'Academic institute'],
        ['laboratories', 'Laboratory'],
        ['laboratory', 'Laboratory'],
        ['corporation', 'Corporation'],
        ['corp', 'Corporation'],
        ['ltd', 'Corporation'],
        ['limited', 'Corporation'],
        ['gmbh', 'Corporation'],
        ['ministry', 'Ministry'],
        ['school of', ''],
    ]
    
    for word, affiliation_type in pairs:
        if word in name:
            return affiliation_type
    
    return 'Unknown'

def clean_source(name):
    return name

def get_affiliations(doc, attribute='name'):
    # Put affiliations of all authors in one list.
    affiliation_lists = [a.affiliations for a in doc.authors]

    # Remove 'None' affialiations
    affiliations = [x for x in affiliation_lists if x is not None]

    # Flatten lists
    affiliations = [y for x in affiliations for y in x]

    if attribute == 'country':
        # Get affiliation countries and remove 'None' values
        affiliations = [af.country for af in affiliations if af.country is not None]
    elif attribute == 'affiliation_type':
        # Get affiliation names
        affiliations = [af.name for af in affiliations]
        affiliations = [affiliation_to_type(x) for x in affiliations]
    else:
        # Get affiliation names
        affiliations = [af.name for af in affiliations]

    # Clean affiliation names
    # affiliations = [clean_affiliation(af) for af in affiliations]

    # Remove duplicates (2 authors with same affiliation/country
    # results in 1 count for that affiliation/country).
    return set(affiliations)

def merge_author_affiliation(doc):
    if doc.authors is None:
        return []

    authors_plus_aff = []
    for author in doc.authors:
        if author.affiliations is None:
            authors_plus_aff.append(author.name)
        else:
            merged = [author.name + ' ' + affiliation.name for affiliation in author.affiliations]
            authors_plus_aff += merged

    return set(authors_plus_aff)

def abbr_to_full_language(language):
    pairs = [
        ['eng', 'English']
    ]
    
    for abbr, full in pairs:
        if language == abbr:
            return full
    
    return language

def plot_year_histogram(docset, ax=None):
    """Plot a histogram of the number of documents published for each year.

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    """

    # Publications per year
    year_count = defaultdict(int)

    for d in docset.docs:
        year_count[d.year] += 1
    
    min_year = min(year_count.keys())
    max_year = max(year_count.keys())
    years = list(range(min_year, max_year+1))
    
    plot_statistic(lambda p: [p.year], docset=docset, x=years, ax=ax, x_label="No. publications")

def plot_author_histogram(docset, top_k=20, ax=None):
    """Plot a histogram of the number of documents published by each author. Note that
    this is done on a best-effort basis since one author could have published under
    multiple spellings of the same name (e.g., "Alan Turing", "A. Turing", or "A. M. Turing")

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    :param top_k: Limit results to the top k authors.
    """
    plot_statistic(lambda p: set(a.name for a in p.authors or []), x=top_k, docset=docset, ax=ax, x_label="No. publications")

def plot_author_affiliation_histogram(docset, top_k=30, ax=None):
    """Plot a histogram of the number of documents published by each combination 
    of author + affiliation. This helps to reduce the name number of collisions for
    different persons having the same name.

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    :param top_k: Limit results to the top k combinations.
    """
    plot_statistic(lambda p: merge_author_affiliation(p), x=top_k, docset=docset, ax=ax, x_label="No. publications")

def plot_number_authors_histogram(docset, ax=None):
    """Plot a histogram of the number of authors per document.

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    """
    plot_statistic(lambda p: [len(set(a.name for a in p.authors or []))], x=range(25), docset=docset, ax=ax, x_label="No. publications")

def plot_source_type_histogram(docset, ax=None):
    """Plot a histogram of the document source types (e.g., Journal, conference proceedings, etc.)

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    """
    plot_statistic(lambda p: [p.source_type], docset=docset, ax=ax, x_label="No. publications")

def plot_source_histogram(docset, x=10, ax=None):
    """Plot a histogram of the document source. Note that this done on a best-effort basis since
    one publication venue could have multiple spellings.

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    :param top_k: Limit results to the top k sources.
    """
    plot_statistic(lambda p: [clean_source(p.source)], x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_affiliation_histogram(docset, x=10, ax=None):
    """Plot a histogram of the number of documents published by each affiliation. Note that
    this is done on a best-effort basis since one affiliation could have multiple spellings
    (e.g., "University of Amsterdam", "Universiteit van Amsterdam", or "UvA").

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    :param top_k: Limit results to the top k institutes.
    """
    plot_statistic(lambda p: get_affiliations(p), x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_country_histogram(docset, top_k=10, ax=None):
    """Plot a histogram of the number of documents published by each country based
    on author affiliation. Note that the country is not always available.

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    :param top_k: Limit results to the top k country.
    """
    plot_statistic(lambda p: get_affiliations(p, attribute='country'), x=top_k, docset=docset, ax=ax, x_label="No. publications")

def plot_affiliation_type_histogram(docset, x=10, ax=None):
    """Plot a histogram of the number of documents published by each type
    of affiliation (e.g., research institutes, academic institute, company, etc.)

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    :param top_k: Limit results to the top k entries.
    """
    plot_statistic(lambda p: get_affiliations(p, attribute='affiliation_type'), x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_language_histogram(docset, ax=None):
    """Plot a histogram of the number of documents published for each language
    (e.g., English, German, Chinese, etc.)

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    """
    plot_statistic(lambda p: [abbr_to_full_language(p.language)], docset=docset, ax=ax, x_label="No. publications")

def plot_words_histogram(freqs, dic, top_k=25, ax=None):
    """Plot a histogram of word frequencies in the documents.

    :param docset: The `DocumentSet`.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    :param top_k: Limit results to the top k entries.
    """
    all_freqs = []
    for doc_freq in freqs:
        all_freqs += doc_freq

    count = defaultdict(int)
    for word, freq in all_freqs:
        count[str(dic[word])] += freq

    plot_statistic(None, docset=None, ax=ax, x_label="No. publications", x=top_k, count=count)


#-----------------------------------------------------------------------
# Wordcloud plotting functions
#-----------------------------------------------------------------------

def plot_topic_clouds(model, cols=3, fig=None, **kwargs):
    """Plot the word distributions of a topic model.

    :param model: The `TopicModel`.
    :param cols: Number of columns (e.g., word clouds per row).
    :param fig: The figure on which to plot the results, defaults to current figure.
    :param \**kwargs: Additional parameters passed to `plot_topic_cloud`.
    """
    if fig is None:
        fig, ax = prepare_fig(2, wordcloud=True)

    rows = math.ceil(model.num_topics / float(cols))

    for i in range(model.num_topics):
        ax = fig.add_subplot(rows, cols, i + 1)
        plot_topic_cloud(model, i, ax=ax, **kwargs)


def plot_topic_cloud(model, topicid, ax=None, **kwargs):
    """Plot the word distributions of a single topic from a topic model.

    :param model: The `TopicModel`.
    :param topicid: The topic index within the topic model.
    :param ax: The axis on which to plot the histogram, defaults to current axis.
    :param \**kwargs: Additional parameters passed to `generate_topic_cloud`.
    """
    if ax is None: ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    im = generate_topic_cloud(model, topicid, **kwargs).to_array()
    ax.imshow(im, interpolation='bilinear')

def generate_topic_cloud(model, topicid, cmap=None, max_font_size=75, background_color='white'):
    """Generate the word cloud for the word distributions of a single topic from a topic model.

    :param model: The `TopicModel`.
    :param topicid: The topic index within the topic model.
    :param cmap: The color map to use for the foreground colors, defaults to "Blues".
    :param background_color: The background color, defaults to "white".
    :param max_font_size: The maximum font size.
    :param \**kwargs: Additional parameters passed to `wordcloud.WordCloud`.
    :return: A `wordcloud.WordCloud` instance.
    """
    if cmap is None: cmap = plt.get_cmap('Blues')

    mapping = dict()
    maximum = np.amax(model.topic2token[topicid])

    for i in np.argsort(model.topic2token[topicid])[-100:]:
        if model.topic2token[topicid, i] > 0:
            mapping[model.dictionary[i]] = model.topic2token[topicid, i] / maximum

    def get_color(word, **kwargs):
        weight = kwargs['font_size'] / 75.0 * 0.7 + 0.3
        r, g, b = np.array(cmap(weight)[:3]) * 255
        return 'rgb({}, {}, {})'.format(int(r), int(g), int(b))

    wc = wordcloud.WordCloud(
            prefer_horizontal=True,
            max_font_size=max_font_size,
            background_color=background_color,
            color_func=get_color,
            scale=2,
            relative_scaling=0.5)
    wc.fit_words(mapping)

    return wc

#-----------------------------------------------------------------------
# Topic map plotting functions
#-----------------------------------------------------------------------

def draw_dot(model, p, t, zorder=0):
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    color = plt.get_cmap('jet')(float(t) / model.num_topics)
    color = 0.8 * np.array(color)[:3]
    
    plt.scatter(
        p[0], 
        p[1],
        s=150,
        c=[color],
        marker='o',
        linewidth=0.5,
        zorder=zorder)
    
    plt.text(
        p[0], 
        p[1],
        labels[t],
        fontsize=6,
        color='1',
        va='center',
        ha='center',
        fontweight='bold',
        zorder=zorder + 1)

def plot_topic_map(model, dic, freqs, fig=None):
    """Embeds the documents onto a 2D plane."""
    seed = 70 # seed for truncatedSVD
    vis_seed = 6 # seed for t-SNE visualization

    tfidf_matrix = create_tfidf(freqs, dic)

    # Lower dimensionality of original frequency matrix to improve cosine distances for visualization
    reduced_matrix = TruncatedSVD(
        n_components=10, 
        random_state=seed
    ).fit_transform(tfidf_matrix)

    # Learn model
    tsne_model = sklearn.manifold.TSNE(
        verbose=True,
        metric='cosine',
        random_state=vis_seed,
        perplexity=20)
    pos = tsne_model.fit_transform(reduced_matrix)

    # Resize so xy-position is between 0.05 and 0.95
    pos -= (np.amin(pos, axis=0) + np.amax(pos, axis=0)) / 2
    pos /= np.amax(np.abs(pos))
    pos = (pos * 0.5) + 0.5
    pos = (pos * 0.9) + 0.05

    if fig is None:
        fig, ax = prepare_fig(2)

    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    zorder = 0

    # Draw dots
    for i in np.random.permutation(len(model.doc2topic)):
        topic_id = np.argmax(model.doc2topic[i])
        draw_dot(model, pos[i], topic_id, zorder)
        zorder += 2

    # Draw legend
    for i in range(model.num_topics):    
        y = 0.95 - i * 0.05
        label = ', '.join(dic[w] for w in np.argsort(model.topic2token[i])[::-1][:3])

        draw_dot(model, [0.015, y], i)
        plt.text(0.03, y, label, ha='left', va='center', fontsize=8, zorder=zorder)
        zorder += 1
