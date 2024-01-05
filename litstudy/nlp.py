from collections import defaultdict
from gensim.matutils import corpus2dense
from typing import List
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wordcloud

from .stopwords import STOPWORDS
from .types import DocumentSet


def filter_tokens(texts, predicate):
    for text in texts:
        yield [token for token in text if predicate(token)]


def preprocess_remove_short(texts, min_length=3):
    yield from filter_tokens(texts, lambda token: len(token) >= min_length)


def preprocess_remove_words(texts, remove_words):
    remove_words = set(w.strip() for w in remove_words)
    yield from filter_tokens(texts, lambda token: token not in remove_words)


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
    phrases = gensim.models.phrases.Phrases(texts, threshold=threshold, scoring="npmi")

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
    """Contains the word-frequency vectors for a set of documents. See
    `build_corpus` for more information.
    """

    def __init__(self, docs, filters, max_tokens):
        corpus = []

        for doc in docs:
            text = (doc.title or "") + " " + (doc.abstract or " ")
            tokens = gensim.utils.tokenize(text, lowercase=True, deacc=True)

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


def build_corpus(
    docs: DocumentSet,
    *,
    remove_words=None,
    min_word_length=3,
    min_docs=5,
    max_docs_ratio=0.75,
    max_tokens=5000,
    replace_words=None,
    custom_bigrams=None,
    ngram_threshold=None
) -> Corpus:
    """Build a `Corpus` object.

    This function takes the words from the title/abstract of the given
    documents, preprocesses the tokens, and returns a corpus consisting of a
    word frequency vector for each document. This preprocessing stage is
    highly customizable, thus it is advised to experiment with the many
    parameters.

    Please notice that a small document set with no Abstracts available, might
    not yield a Corpus, since there is a higher chance of words not achieving
    a ocorrency in more than one document.

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
        filters.append(lambda w: preprocess_remove_short(w, min_length=min_word_length))

    filters.append(preprocess_stopwords)

    if ngram_threshold is not None:
        filters.append(lambda w: preprocess_merge_ngrams(w, ngram_threshold))

    filters.append(preprocess_smart_stemming)

    if min_docs > 1 or max_docs_ratio < 1.0:
        max_docs = int(len(docs) * max_docs_ratio)
        filters.append(lambda w: preprocess_outliers(w, min_docs, max_docs))

    return Corpus(docs, filters, max_tokens)


class TopicModel:
    """Topic model trained by one of the `train_*_model` functions."""

    def __init__(self, dictionary, doc2topic, topic2token):
        self.dictionary = dictionary

        self.doc2topic = doc2topic
        """ `N x T` matrix that stores the weights towards each of the T
        topics for the N documents.
        """

        self.topic2token = topic2token
        """ `T x M` matrix that stores the weights towards each of the M
        tokens for each of the T topics
        """

        self.num_topics = len(topic2token)

    def best_documents_for_topic(self, topic_id: int, limit=5) -> List[int]:
        """Returns the documents that most strongly belong to the given
        topic.
        """
        return np.argsort(self.doc2topic[:, topic_id])[::-1][:limit]

    def document_topics(self, doc_id: int):
        """Returns a numpy array indicating the weights towards the different
        topic for the given document. These weight sum up to one.
        """
        return self.doc2topic[doc_id]

    def best_token_weights_for_topic(self, topic_id: int, limit=5):
        """Returns a list of `(token, weight)` tuples for the tokens that
        most strongly belong to the given topic.
        """
        dic = self.dictionary
        weights = self.topic2token[topic_id]

        indices = np.argsort(self.topic2token[topic_id])[::-1][:limit]
        return [(dic[i], weights[i]) for i in indices]

    def best_tokens_for_topic(self, topic_id: int, limit=5):
        """Returns the top tokens that most strongly belong to the given
        topic."""
        results = self.best_token_weights_for_topic(topic_id, limit=limit)
        return [w for w, _ in results]

    def best_token_for_topic(self, topic_id: int) -> str:
        """Returns the token that most strongly belongs to the given
        topic."""
        return self.best_tokens_for_topic(topic_id, limit=1)[0]

    def best_topic_for_token(self, token) -> int:
        """Returns the topic index that most strongly belongs to the given
        token."""
        index = self.dictionary.token2id[token]
        return np.argmax(self.topic2token[:, index])

    def best_topic_for_documents(self) -> List[int]:
        """Returns the topic for each document that most strongly belongs
        to that document.
        """
        return np.argmax(self.doc2topic, axis=1)


def train_nmf_model(corpus: Corpus, num_topics: int, seed=0, max_iter=500) -> TopicModel:
    """Train a topic model using NMF.

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
        h_max_iter=50,
    )

    doc2topic = corpus2dense(model[freqs], num_topics).T
    topic2token = model.get_topics()

    return TopicModel(dic, doc2topic, topic2token)


def train_lda_model(corpus: Corpus, num_topics, seed=0, **kwargs) -> TopicModel:
    """Train a topic model using LDA.

    :param num_topics: The number of topics to train.
    :param seed: The seed used for random number generation.
    :param kwargs: Arguments passed to `gensim.models.lda.LdaModel` (gensim3)
                   or `gensim.models.ldamodel.LdaModel` (gensim4).
    """

    dic = corpus.dictionary
    freqs = corpus.frequencies

    from importlib.metadata import version
    gensim_mayor=version('gensim').split('.')[0]

    if gensim_mayor == 3:
        from gensim.models.lda import LdaModel
        model = LdaModel(list(corpus), **kwargs)
    elif gensim_mayor == 4:
        from gensim.models.ldamodel import LdaModel
        model = LdaModel(freqs, id2word=dic, **kwargs)
    else:
        sys.exit('LdaModel could not be imported from gensim 3 or 4.')

    doc2topic = corpus2dense(model[freqs], num_topics)
    topic2token = model.get_topics()

    return TopicModel(dic, doc2topic, topic2token)


def compute_word_distribution(corpus: Corpus, *, limit=None) -> pd.DataFrame:
    """Returns dataframe that indicates, for each word, the number of
    documents that mention that word.
    """
    counter = defaultdict(int)
    dic = corpus.dictionary

    for vector in corpus.frequencies:
        for i, _ in vector:
            counter[i] += 1

    keys = sorted(counter, key=lambda k: counter[k], reverse=True)[:limit]
    if limit is not None:
        keys = keys[:limit]

    return pd.DataFrame(index=[dic[i] for i in keys], data=dict(count=[counter[i] for i in keys]))


def generate_topic_cloud(
    model: TopicModel, topic_id: int, cmap=None, max_font_size=75, background_color="white"
) -> wordcloud.WordCloud:
    """Generate a word cloud for the given topic from the given topic model.

    :param cmap: The color map used to color the words.
    :param max_font_size: Size of the word which most strongly belongs to the
                          topic. The other words are scaled accordingly.
    :param background_color: Background color.
    """
    if cmap is None:
        cmap = "Blues"

    cmap = plt.get_cmap(cmap)

    dic = model.dictionary
    vec = model.topic2token[topic_id]
    maximum = np.amax(vec)
    words = dict()

    for i in np.argsort(vec)[-100:]:
        if vec[i] > 0:
            words[dic[i]] = vec[i] / maximum

    def get_color(word, **kwargs):
        weight = kwargs["font_size"] / 75.0 * 0.7 + 0.3
        r, g, b = np.array(cmap(weight)[:3]) * 255
        return "rgb({}, {}, {})".format(int(r), int(g), int(b))

    wc = wordcloud.WordCloud(
        prefer_horizontal=True,
        max_font_size=max_font_size,
        background_color=background_color,
        color_func=get_color,
        scale=2,
        relative_scaling=0.5,
    )

    wc.fit_words(words)

    return wc


def calculate_embedding(corpus: Corpus, *, rank=2, svd_dims=50, perplexity=30, seed=0):
    """Calculate a document embedding that assigns each document in the
    corpus a N-d position based on the word usage.

    :returns: A list of N-d tuples for the documents in the corpus.
    """
    from gensim.models.tfidfmodel import TfidfModel
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    dic = corpus.dictionary
    freqs = corpus.frequencies
    tfidf = corpus2dense(TfidfModel(dictionary=dic)[freqs], len(dic)).T

    if svd_dims is not None:
        svd = TruncatedSVD(n_components=svd_dims, random_state=seed)
        components = svd.fit_transform(tfidf)
    else:
        components = tfidf

    model = TSNE(rank, metric="cosine", perplexity=perplexity, random_state=seed)
    return model.fit_transform(components)
