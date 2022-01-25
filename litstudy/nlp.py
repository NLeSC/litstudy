from collections import defaultdict
from gensim.matutils import corpus2dense
import gensim
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import wordcloud

from .plot import plot_histogram
from .stopwords import STOPWORDS
from .types import DocumentSet


def filter_tokens(texts, predicate):
    for text in texts:
        yield [token for token in text if predicate(token)]


def preprocess_remove_short(texts, min_length=3):
    yield from filter_tokens(texts, lambda token: len(token) >= min_length)


def preprocess_remove_words(texts, stopwords):
    stopwords = set(w.strip() for w in stopwords)
    yield from filter_tokens(texts, lambda token: token not in STOPWORDS)


def preprocess_stopwords(texts):
    yield from preprocess_remove_words(texts, STOPWORDS)


def preprocess_replace_words(texts, replace_words):
    for text in texts:
        yield [replace_words.get(token, token) for token in text]


def preprocess_merge_bigrams(texts, bigrams):
    for text in texts:
        prev = None
        new_text = []

        for current in text:
            replacement = bigrams.get((prev, current))

            if replacement is not None:
                new_text.append(replacement)
                prev = None
            else:
                if prev is not None:
                    new_text.append(prev)
                prev = current

        yield new_text


def preprocess_merge_ngrams(texts, threshold):
    texts = list(texts)
    phrases = gensim.models.phrases.Phrases(texts, threshold=threshold,
                                            scoring='npmi')

    for text in texts:
        for word, score in phrases.analyze_sentence(text):
            if score is not None:
                text.append(word)

        yield text


def preprocess_outliers(texts, min_docs, max_docs):
    texts = list(texts)
    count = defaultdict(int)

    for text in texts:
        for token in set(text):
            count[token] += 1

    unwanted = set()
    for token, num_docs in count.items():
        if num_docs < min_docs or num_docs > max_docs:
            unwanted.add(token)

    yield from preprocess_remove_words(texts, unwanted)


def preprocess_smart_stemming(texts):
    texts = list(texts)
    stemmer = gensim.parsing.PorterStemmer()
    count = defaultdict(int)

    for text in texts:
        for token in text:
            count[token] += 1

    sorted_count = sorted(count.items(), key=lambda p: p[1], reverse=True)
    unstemming = dict()
    mapping = dict()

    for token, _ in sorted_count:
        stem = stemmer.stem(token)
        if stem in unstemming:
            mapping[token] = unstemming[stem]
        else:
            unstemming[stem] = token
            mapping[token] = token

    for text in texts:
        yield [mapping[token] for token in text]


class Corpus:
    """ Contains the word-frequency vectors for a set of documents. See
        `build_corpus` for more information.
    """
    def __init__(self, docs, filters, max_tokens):
        corpus = []

        for doc in docs:
            text = (doc.title or '') + ' ' + (doc.abstract or ' ')
            tokens = gensim.utils.tokenize(
                    text,
                    lowercase=True,
                    deacc=True)

            corpus.append(list(tokens))

        for f in filters:
            corpus = f(corpus)

        corpus = list(corpus)
        dic = gensim.corpora.Dictionary(corpus)
        dic.filter_extremes(keep_n=max_tokens)

        self.dictionary = dic
        """ The dictionary that maps indices to words
            (`gensim.corpora.Dictionary`).
        """

        self.frequencies = [dic.doc2bow(x) for x in corpus]
        """ List of word frequency vectors. Each vector corresponds to one
        document and consists of `(word_index, frequency)` tuples.
        """


def build_corpus(docs: DocumentSet, *, remove_words=None, min_word_length=3,
                 min_docs=5, max_docs_ratio=0.75, max_tokens=5000,
                 replace_words=None, custom_bigrams=None, ngram_threshold=None
                 ) -> Corpus:
    """ Build a `Corpus` object.

    This function takes the words from the title/abstract of the given
    documents, preprocesses the tokens, and returns a corpus consisting of a
    word frequency vector for each document. This preprocessing stage is
    highly customizable, thus it is advised to experiment with the many
    parameters.

    :param remove_words: list of words that should be ignored while building
                         the word frequency vectors.
    :param min_word_length: Words shorter than this are ignored.
    :param min_docs: Words that occur in fewer than this many documents are
                     ignored.
    :param max_docs_ratio: Words that occur in more than this document are
                           ignored. Should be ratio between 0 and 1.
    :param max_tokens: Only the top most common tokens are preserved.
    :param replace_words: Replace words by other words. Must be a `dict`
                          containing *original word* to *replacement word*
                          pairs.
    :param custom_bigrams: Add custom bigrams. Must be a `dict` where keys
                           are `(first, second)` tuples and values are
                           replacements. For example, the key can be
                           `("Big", "Data")` and the value `"BigData"`.
    :param ngram_threshold: Threshold used for n-gram detection. Is passed
                            to `gensim.models.phrases.Phrases` to detect
                            common n-grams.
    :returns: a `Corpus object`.
    """

    filters = []
    if custom_bigrams:
        filters.append(lambda w: preprocess_merge_bigrams(w, custom_bigrams))

    if remove_words:
        filters.append(lambda w: preprocess_remove_words(w, remove_words))

    if replace_words:
        filters.append(lambda w: preprocess_replace_words(w, replace_words))

    if min_word_length:
        filters.append(lambda w: preprocess_remove_short(w,
                       min_length=min_word_length))

    filters.append(preprocess_stopwords)

    if ngram_threshold is not None:
        filters.append(lambda w: preprocess_merge_ngrams(w, ngram_threshold))

    filters.append(preprocess_smart_stemming)

    if min_docs > 1 or max_docs_ratio < 1.0:
        max_docs = int(len(docs) * max_docs_ratio)
        filters.append(lambda w: preprocess_outliers(w, min_docs, max_docs))


    return Corpus(docs, filters, max_tokens)


class TopicModel:
    """ Topic model trained by one of the `train_*_model` functions. """

    def __init__(self, dictionary, doc2topic, topic2token):
        self.dictionary = dictionary
        self.doc2topic = doc2topic
        self.topic2token = topic2token
        self.num_topics = len(topic2token)

    def top_documents_for_topic(self, topic_id, limit=5):
        return np.argsort(self.doc2topic[:,topic_id])[::-1][:limit]

    def document_topics(self, doc_id):
        return self.doc2topic[doc_id]

    def document_topic_weight(self, doc_id, topic_id):
        return self.doc2topic[doc_id, topic_id]

    def top_topic_tokens_with_weights(self, topic_id, limit=5):
        dic = self.dictionary
        weights = self.topic2token[topic_id]

        indices = np.argsort(self.topic2token[topic_id])[::-1][:limit]
        return [(dic[i], weights[i]) for i in indices]

    def top_topic_tokens(self, topic_id, limit=5):
        results = self.top_topic_tokens_with_weights(topic_id, limit=limit)
        return [w for w, _ in results]

    def top_topic_token(self, topic_id):
        return self.top_topic_tokens(topic_id, limit=1)[0]

    def best_topic_for_token(self, token):
        index = self.dictionary.token2id[token]
        return np.argmax(self.topic2token[:, index])


def train_nmf_model(corpus: Corpus, num_topics: int, seed=0, max_iter=500
                    ) -> TopicModel:
    """ Train a topic model using NMF.

    :param num_topics: The number of topics to train.
    :param seed: The seed used for random number generation.
    :param max_iter: The maximum number of iterations to use for training.
                     More iterations mean better results, but longer training
                     times.
    """
    import gensim.models.nmf

    dic = corpus.dictionary
    freqs = corpus.frequencies

    tfidf = gensim.models.tfidfmodel.TfidfModel(dictionary=dic)
    model = gensim.models.nmf.Nmf(
            list(tfidf[freqs]),
            num_topics=num_topics,
            passes=max_iter,
            random_state=seed,
            w_stop_condition=1e-9,
            h_stop_condition=1e-9,
            w_max_iter=50,
            h_max_iter=50)

    doc2topic = corpus2dense(model[freqs], num_topics).T
    topic2token = model.get_topics()

    return TopicModel(dic, doc2topic, topic2token)


def train_lda_model(corpus: Corpus, num_topics, seed=0, **kwargs
                    ) -> TopicModel:
    """ Train a topic model using LDA.

    :param num_topics: The number of topics to train.
    :param seed: The seed used for random number generation.
    :param kwargs: Arguments passed to `gensim.models.lda.LdaModel`.
    """
    from gensim.models.lda import LdaModel

    dic = corpus.dictionary
    freqs = corpus.frequencies

    model = LdaModel(list(corpus), **kwargs)

    doc2topic = corpus2dense(model[freqs], num_topics)
    topic2token = model.get_topics()

    return TopicModel(dic, doc2topic, topic2token)


def compute_word_distribution(corpus, *, limit=None):
    """ """
    counter = defaultdict(int)
    dic = corpus.dictionary

    for vector in corpus.frequencies:
        for i, _ in vector:
            counter[i] += 1

    keys = sorted(counter, key=lambda k: counter[k], reverse=True)[:limit]
    if limit is not None:
        keys = keys[:limit]

    return pd.DataFrame(
            index=[dic[i] for i in keys],
            data=dict(count=[counter[i] for i in keys])
    )

def plot_word_distribution(corpus, *, limit=25, **kwargs):
    """ """
    n = len(corpus.frequencies)
    data = compute_word_distribution(corpus, limit=limit)
    return plot_histogram(data, relative_to=n, **kwargs)


def plot_topic_clouds(model: TopicModel, fig=None, ncols=3, **kwargs):
    """ """
    if fig is None:
        plt.clf()
        fig = plt.gcf()

    nrows = math.ceil(model.num_topics / float(ncols))

    for i in range(model.num_topics):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_title(f'Topic {i + 1}')
        plot_topic_cloud(model, i, ax=ax, **kwargs)


def plot_topic_cloud(model: TopicModel, topic_id, ax=None, **kwargs):
    """ """
    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])

    im = generate_topic_cloud(model, topic_id, **kwargs).to_array()
    ax.imshow(im, interpolation='bilinear')


def generate_topic_cloud(model, topic_id, cmap=None, max_font_size=75,
                         background_color='white'):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    dic = model.dictionary
    vec = model.topic2token[topic_id]
    maximum = np.amax(vec)
    words = dict()

    for i in np.argsort(vec)[-100:]:
        if vec[i] > 0:
            words[dic[i]] = vec[i] / maximum

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

    wc.fit_words(words)

    return wc


def calculate_embedding(corpus, rank=2):
    from gensim.models.tfidfmodel import TfidfModel
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    dic = corpus.dictionary
    freqs = corpus.frequencies
    tfidf = corpus2dense(TfidfModel(dictionary=dic)[freqs], len(dic)).T

    svd = TruncatedSVD(n_components=50)
    components = svd.fit_transform(tfidf)
    return TSNE(rank, metric='cosine').fit_transform(components)


def plot_embedding(corpus, model, layout=None, ax=None):
    """ """
    if ax is None:
        ax = plt.gca()

    if layout is None:
        layout = calculate_embedding(corpus)

    dic = corpus.dictionary
    freqs = corpus2dense(corpus.frequencies, len(dic))

    num_topics = len(model.topic2token)
    best_topic = np.argmax(model.doc2topic.T, axis=0)

    colors = seaborn.color_palette('hls', num_topics)
    colors = np.array(colors)[:, :3] * 0.9  # Mute colors a bit

    for i in range(num_topics):
        indices = best_topic == i
        #label = 'ABCDEFGHIJLMNOPQRSTUVWXYZ'[i]
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

        top_items = np.argsort(model.topic2token[i])[::-1]
        label = f'Topic {label}:' + ', '.join(dic[j] for j in top_items[:3])

        center = np.median(layout[indices], axis=0)
        ax.text(
                center[0],
                center[1],
                label,
                va='center',
                ha='center',
                color='1',
                backgroundcolor=(0, 0, 0, .75),
                zorder=10 * len(freqs),
        )

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
