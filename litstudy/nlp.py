from collections import defaultdict
from gensim.matutils import corpus2dense
import gensim
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import wordcloud

from .plot import plot_histogram
from .stopwords import STOPWORDS


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
    phrases = gensim.models.phrases.Phrases(texts, threshold=threshold)
    return phrases[texts]


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

        self.corpus = corpus
        self.dictionary = dic
        self.frequencies = [dic.doc2bow(x) for x in corpus]


def build_corpus(docs, *, remove_words=None, min_word_length=3, min_docs=5,
                 max_docs_ratio=0.75, max_tokens=5000, replace_words=None,
                 custom_bigrams=None, ngram_threshold=None):

    filters = []
    if custom_bigrams:
        filters.append(lambda w: preprocess_merge_bigrams(w, custom_bigrams))

    if remove_words:
        filters.append(lambda w: preprocess_remove_words(w, remove_words))

    if ngram_threshold is not None:
        filters.append(lambda w: preprocess_merge_ngrams(w, ngram_threshold))

    if replace_words:
        filters.append(lambda w: preprocess_replace_words(w, replace_words))

    if min_word_length:
        filters.append(lambda w: preprocess_remove_short(w,
                       min_length=min_word_length))

    if min_docs > 1 or max_docs_ratio < 1.0:
        max_docs = int(len(docs) * max_docs_ratio)
        filters.append(lambda w: preprocess_outliers(w, min_docs, max_docs))

    filters.append(preprocess_stopwords)
    filters.append(preprocess_smart_stemming)

    return Corpus(docs, filters, max_tokens)


def plot_word_distribution(corpus, top=25, **kwargs):
    counter = defaultdict(int)
    dic = corpus.dictionary
    n = len(corpus.frequencies)

    for vector in corpus.frequencies:
        for i, freq in vector:
            counter[i] += 1

    best = sorted(counter, key=lambda k: counter[k], reverse=True)[:top]
    keys = [dic[i] for i in best]
    values = [counter[i] for i in best]

    return plot_histogram(keys, values, relative_to=n, **kwargs)


class TopicModel:
    def __init__(self, dictionary, doc2topic, topic2token):
        self.dictionary = dictionary
        self.doc2topic = doc2topic
        self.topic2token = topic2token
        self.num_topics = len(topic2token)


def train_nmf_model(corpus, num_topics, seed=0, max_iter=500):
    import gensim.models.nmf

    dic = corpus.dictionary
    freqs = corpus.frequencies

    tfidf = gensim.models.tfidfmodel.TfidfModel(dictionary=dic)

    for n in range(1, 100):
        errors = []

        for seed in [0, 1, 2, 3, 4]:
            model = gensim.models.nmf.Nmf(
                    list(tfidf[freqs]),
                    # num_topics=num_topics,
                    num_topics=n,
                    passes=max_iter,
                    random_state=seed,
                    w_stop_condition=1e-9,
                    h_stop_condition=1e-9,
                    w_max_iter=50,
                    h_max_iter=50)
            errors.append(model._w_error)

    doc2topic = corpus2dense(model[freqs], num_topics)
    topic2token = model.get_topics()

    return TopicModel(dic, doc2topic, topic2token)


def train_lda_model(corpus, num_topics, seed=0, **kwargs):
    from gensim.models.lda import LdaModel

    dic = corpus.dictionary
    freqs = corpus.frequencies

    model = LdaModel(list(corpus), **kwargs)

    doc2topic = corpus2dense(model[freqs], num_topics)
    topic2token = model.get_topics()

    return TopicModel(dic, doc2topic, topic2token)


def plot_topic_clouds(model, fig=None, ncols=3, **kwargs):
    if fig is None:
        plt.clf()
        fig = plt.gcf()

    nrows = math.ceil(model.num_topics / float(ncols))

    for i in range(model.num_topics):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot_topic_cloud(model, i, ax=ax, **kwargs)


def plot_topic_cloud(model, topic_id, ax=None, **kwargs):
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
    if ax is None:
        ax = plt.gca()

    if layout is None:
        layout = calculate_embedding(corpus)

    dic = corpus.dictionary
    freqs = corpus2dense(corpus.frequencies, len(dic))

    num_topics = len(model.topic2token)
    best_topic = np.argmax(model.doc2topic, axis=0)

    colors = seaborn.color_palette('hls', num_topics)
    colors = np.array(colors)[:, :3] * 0.9  # Mute colors a bit

    for i in range(num_topics):
        indices = best_topic == i
        letter = 'ABCDEFGHIJLMNOPQRSTUVWXYZ'[i]

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
                letter,
                fontsize=6,
                color='1',
                va='center',
                ha='center',
                fontweight='bold',
                zorder=2*j + 1,
            )

        top_items = np.argsort(model.topic2token[i])[::-1]
        label = f'Topic {letter}:' + ', '.join(dic[j] for j in top_items[:3])

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
