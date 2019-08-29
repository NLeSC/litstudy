import numpy as np
import gensim
import gensim.models.nmf
import sys
import sklearn.feature_extraction.text
import sklearn.decomposition
import wordcloud
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from sklearn.decomposition import TruncatedSVD

import sklearn.manifold
import sklearn.metrics.pairwise

def prepare_fig(w=1, h=None):
    if h is None: h = w
    return plt.figure(figsize=(6 * w, 3 * h))

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

def plot_topic_distribution(model, dic, freqs, fig=None):
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
        fig = prepare_fig(2)#plt.gcf()
        ax = plt.gca()

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
        y = 0.985 - i * 0.02
        label = ', '.join(dic[w] for w in np.argsort(model.topic2token[i])[::-1][:3])

        draw_dot(model, [0.015, y], i)
        plt.text(0.03, y, label, ha='left', va='center', fontsize=8, zorder=zorder)
        zorder += 1

def plot_topic_clouds(model, cols=3, fig=None, **kwargs):
    if fig is None:
        fig = prepare_fig(2)#plt.gcf()
        ax = plt.gca()

    rows = int(model.num_topics / float(cols) + cols - 1)

    for i in range(model.num_topics):
        ax = fig.add_subplot(rows, cols, i + 1)
        plot_topic_cloud(model, i, ax=ax, **kwargs)


def plot_topic_cloud(model, topicid, ax=None, **kwargs):
    if ax is None: ax = plt.gca()

    im = generate_topic_cloud(model, topicid, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(generate_topic_cloud(model, topicid), interpolation='bilinear')

def generate_topic_cloud(model, topicid, cmap=None, max_font_size=75, background_color='white'):
    if cmap is None: 
        cmap = plt.get_cmap('Blues')

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

    return wc.to_array()

class Corpus:
    def __init__(self, texts):
        self.texts = texts

class TopicModel:
    def __init__(self, corpus, doc2topic, topic2token):
        self.corpus = corpus
        self.dictionary = corpus
        self.doc2topic = doc2topic
        self.topic2token = topic2token
        self.num_topics = len(topic2token)
        self.num_documents = len(doc2topic)
        self.num_tokens = len(topic2token.T)

def create_tfidf(freqs, dic):
    # Build dense matrix
    matrix = np.zeros((len(freqs), len(dic)))
    for i, vec in enumerate(freqs):
        for j, f in vec:
            matrix[i, j] = f

    # Apply TFIDF
    tfidf_model = sklearn.feature_extraction.text.TfidfTransformer()
    tfidf_matrix = tfidf_model.fit_transform(matrix).toarray()

    return tfidf_matrix

def train_nmf_model(dic, freqs, num_topics, seed=0, **kwargs):
    tfidf_matrix = create_tfidf(freqs, dic)

    # Train NMF model
    nmf_model = sklearn.decomposition.NMF(
	n_components=num_topics,
	random_state=seed,
	tol=1e-9,
	max_iter=500,
        **kwargs)

    # Train model
    doc2topic = nmf_model.fit_transform(tfidf_matrix)
    topic2token = nmf_model.components_

    # Normalize token distributions.
    sums = np.sum(topic2token, axis=1) # k
    topic2token /= sums.reshape(-1, 1) # k,m
    doc2topic *= sums.reshape(1, -1) # n,k

    # Normalize topic distributions.
    doc2topic /= np.sum(doc2topic, axis=1).reshape(-1, 1) # n,k

    return TopicModel(dic, doc2topic, topic2token)

def train_lda_model(dic, freqs, num_topics, **kwargs):
    model = gensim.models.LdaModel(freqs, num_topics, id2word=dic, **kwargs)
    topic2token = model.get_topics()
    doc2topic = np.zeros((len(freqs), num_topics))

    for i in range(len(freqs)):
        for j, f in model.get_document_topics(freqs[i]):
            doc2topic[i, j] = f
            
    return TopicModel(dic, doc2topic, topic2token)


def build_corpus_simple(docs, stopwords=[], bigrams={}, min_length=2):
    filters = [
        partial(merge_bigrams, bigrams=bigrams),
        strip_default_stopwords,
        partial(strip_stopwords, stopwords=set(stopwords)),
        partial(strip_short, min_length=min_length),
        stem_smart,
    ]

    return build_corpus(docs, filters)

def build_corpus(docs, filters):
    corpus = []

    for doc in docs:
        tokens = gensim.utils.tokenize(
                (doc.title or ' ') + ' ' + (doc.abstract or ' '),
                lowercase=True,
                deacc=True)

        corpus.append(tokens)

    for f in filters:
        corpus = f(corpus)

    corpus = list(map(list, corpus))
    dic = gensim.corpora.Dictionary(corpus)
    freqs = [dic.doc2bow(x) for x in corpus]

    return dic, freqs

def merge_bigrams(texts, bigrams):
    # TODO
    return texts

def strip_short(texts, min_length=3):
    for text in texts:
        yield [token for token in text if len(token) >= min_length]

def strip_numeric(texts):
    return map(gensim.parsing.preprocessing.strip_numeric, texts)

def strip_non_alphanum(texts):
    return map(gensim.parsing.preprocessing.strip_non_alphanum, texts)

def strip_stopwords(texts, stopwords):
    for text in texts:
        yield [token for token in text if token not in stopwords]

def strip_default_stopwords(texts):
    return strip_stopwords(texts, gensim.parsing.preprocessing.STOPWORDS)

def stem_porter(texts):
    stemmer = gensim.parsing.PorterStemmer()
    for text in texts:
        yield [stemmer.stem(token) for token in text]

def stem_smart(texts):
    texts = list(texts)
    stemmer = gensim.parsing.PorterStemmer()
    word_count = defaultdict(int)
    stemming = dict()
    unstemming = dict()

    for text in texts:
        for token in text:
            word_count[token] += 1

    for token, _ in sorted(word_count.items(), key=lambda p: p[1]):
        stem = stemmer.stem(token)
        stemming[token] = stem
        unstemming[stem] = token

    for text in texts:
        yield [unstemming[stemming[token]] for token in text]
        
