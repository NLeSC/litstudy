import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

def top_k(mapping, k=10):
    return sorted(mapping.keys(), key=lambda x: mapping[x])[::-1][:k]

def prepare_fig(w=1, h=None):
    if h is None: h = w
    return plt.figure(figsize=(6 * w, 3 * h))
    # sns.set(rc={'figure.figsize': figsize})
    # plt.clf()

# Publications per aggregation type
def plot_statistic(fun, docset, x=None, ax=None, x_label=""):
    """ Plot statistics of some sort in a bar plot. If x is not given,
        (None) all keys with a count > 0 are plotted. If x is a list, the
        counts of all list elements are included. If x is an integer,
        the x keys with highest counts are plotted. """

    if ax is None:
        fig = prepare_fig(2)
        ax = plt.gca()

    count = defaultdict(int)

    for d in docset.docs:
        for key in fun(d):
            if key:
                # count[unicode(key)] += 1
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

def clean_source(name):
    return name

def get_affiliations(doc):
    # Put affiliations of all authors in one list.
    affiliation_lists = [a.affiliations for a in doc.authors]

    # Remove 'None' affialiations
    affiliations = [x for x in affiliation_lists if x is not None]

    # Flatten lists
    affiliations = [y for x in affiliations for y in x]

    # Get affiliation names
    affiliations = [af.name for af in affiliations]
    
    # Clean affiliation names
    # affiliations = [clean_affiliation(af) for af in affiliations]

    # Remove duplicates (2 authors with same affiliation results in 1
    # count for that affiliation).
    return set(affiliations)

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

def plot_author_histogram(docset, ax=None):
    plot_statistic(lambda p: set(a.name for a in p.authors or []), x=5, docset=docset, ax=ax, x_label="No. publications")

def plot_number_authors_histogram(docset, ax=None):
    plot_statistic(lambda p: [len(set(a.name for a in p.authors or []))], x=5, docset=docset, ax=ax, x_label="No. publications")

def plot_source_type_histogram(docset, ax=None):
    plot_statistic(lambda p: [p.source_type], docset=docset, ax=ax, x_label="No. publications")

def plot_source_histogram(docset, ax=None):
    plot_statistic(lambda p: [clean_source(p.source)], x=10, docset=docset, ax=ax, x_label="No. publications")

def plot_source_histogram(docset, ax=None):
    plot_statistic(lambda p: [p.source], docset=docset, ax=ax, x_label="No. publications")

def plot_affiliation_histogram(docset, ax=None):
    # Publications per institute
    plot_statistic(lambda p: get_affiliations(p), x=10, docset=docset, ax=ax, x_label="No. publications")
