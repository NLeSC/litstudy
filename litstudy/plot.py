import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd

def top_k(mapping, k=10):
    return sorted(mapping.keys(), key=lambda x: mapping[x])[::-1][:k]

def prepare_fig(w=1, h=None):
    if h is None: h = w
    return plt.figure(figsize=(6 * w, 3 * h))
    # sns.set(rc={'figure.figsize': figsize})
    # plt.clf()

# Publications per aggregation type
def plot_statistic(fun, docset, x=None, ax=None, x_label="", count=None):
    """ Plot statistics of some sort in a bar plot. If x is not given,
        (None) all keys with a count > 0 are plotted. If x is a list, the
        counts of all list elements are included. If x is an integer,
        the x keys with highest counts are plotted. """

    if ax is None:
        fig = prepare_fig(2)
        ax = plt.gca()

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

    # prepare_fig(1, 4)
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
    # print("A")
    # print(authors_plus_aff)
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
    # Publications per year
    year_count = defaultdict(int)

    for d in docset.docs:
        year_count[d.year] += 1
    
    min_year = min(year_count.keys())
    max_year = max(year_count.keys())

    # years = range(2000, 2020)
    years = list(range(min_year, max_year+1))
    
    # plot_statistic(lambda p: [p.year], docset=docset, x=years, ax=ax, x_label="No. publications")
    plot_statistic(lambda p: [p.year], docset=docset, x=years, ax=ax, x_label="No. publications")

def plot_author_histogram(docset, x=20, ax=None):
    plot_statistic(lambda p: set(a.name for a in p.authors or []), x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_author_affiliation_histogram(docset, x=30, ax=None):
    plot_statistic(lambda p: merge_author_affiliation(p), x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_number_authors_histogram(docset, x=5, ax=None):
    plot_statistic(lambda p: [len(set(a.name for a in p.authors or []))], x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_source_type_histogram(docset, ax=None):
    plot_statistic(lambda p: [p.source_type], docset=docset, ax=ax, x_label="No. publications")

def plot_source_histogram(docset, x=10, ax=None):
    plot_statistic(lambda p: [clean_source(p.source)], x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_affiliation_histogram(docset, x=10, ax=None):
    # Publications per institute
    plot_statistic(lambda p: get_affiliations(p), x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_country_histogram(docset, x=10, ax=None):
    # Publications per institute
    plot_statistic(lambda p: get_affiliations(p, attribute='country'), x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_affiliation_type_histogram(docset, x=10, ax=None):
    # Publications per institute
    plot_statistic(lambda p: get_affiliations(p, attribute='affiliation_type'), x=x, docset=docset, ax=ax, x_label="No. publications")

def plot_language_histogram(docset, ax=None):
    # Publications per institute
    plot_statistic(lambda p: [abbr_to_full_language(p.language)], docset=docset, ax=ax, x_label="No. publications")

def plot_words_histogram(freqs, dic, x=25, ax=None):
    all_freqs = []
    for doc_freq in freqs:
        all_freqs += doc_freq

    count = defaultdict(int)
    for word, freq in all_freqs:
        count[str(dic[word])] += freq

    plot_statistic(None, docset=None, ax=ax, x_label="No. publications", x=x, count=count)

    # display(pd.DataFrame(
    #     # [(w, word_count[w], 'Yes' * (w in stopwords)) for w in top_k(one_count, 250)],
    #     [(w, count[w]) for w in top_k(count, 250)],
    #     columns=['word', 'count']))