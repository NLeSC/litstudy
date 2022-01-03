from .sources.scopus import search_scopus, refine_scopus
from .sources.bibtex import load_bibtex
from .sources.dblp import search_dblp
from .sources.crossref import search_crossref, refine_crossref
from .sources.semanticscholar import \
        search_semanticscholar, \
        refine_semanticscholar
from .stats import \
        compute_year_histogram, \
        compute_author_histogram, \
        compute_author_affiliation_histogram, \
        compute_language_histogram, \
        compute_number_authors_histogram, \
        compute_source_histogram, \
        compute_affiliation_histogram, \
        compute_country_histogram
from .nlp import \
        build_corpus, \
        train_nmf_model, \
        train_lda_model, \
        plot_topic_clouds, \
        plot_word_distribution, \
        calculate_embedding, \
        plot_embedding
from .network import \
        build_citation_network, \
        build_coauthor_network, \
        build_cocitation_network, \
        build_coupling_network, \
        plot_citation_network, \
        plot_coauthor_network, \
        plot_cocitation_network, \
        plot_coupling_network, \
        plot_network
from .plot import \
        plot_year_histogram, \
        plot_author_histogram, \
        plot_number_authors_histogram, \
        plot_author_affiliation_histogram, \
        plot_language_histogram, \
        plot_source_histogram, \
        plot_affiliation_histogram, \
        plot_country_histogram


__all__ = [
    'network',
    'plot',
    'stats',
]
