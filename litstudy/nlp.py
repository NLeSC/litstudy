import numpy as np
import gensim
import gensim.models.nmf
import sys
import sklearn.feature_extraction.text
import wordcloud
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from sklearn.decomposition import TruncatedSVD

import sklearn.manifold
import sklearn.metrics.pairwise

# def trans(tfidf_matrix):
#     new_tfidf_matrix = []

#     for i, doc in enumerate(tfidf_matrix):
#         new_tfidf_matrix.append(defaultdict(int))
#             for word, tfidf in tfidf_matrix[i]:
#                 new_tfidf_matrix[i][word] = tfidf

#     return tfidf_matrix

def prepare_fig(w=1, h=None):
    if h is None: h = w
    return plt.figure(figsize=(6 * w, 3 * h))

def draw_dot(p, t, zorder=0):
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    num_topics = 4

    color = plt.get_cmap('jet')(float(t) / num_topics)
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
    seed = 70 # seed for NMF topic model
    vis_seed = 6 # seed for t-SNE visualization
    vis_angle = 135 # rotation angle for visualization
    num_topics = model.num_topics

    # Create frequency matrix
    n, m = len(freqs), len(dic)
    matrix = np.zeros((n, m))

    for i, row in enumerate(freqs):
        for j, freq in row:
            matrix[i, j] = freq
            
    # Run TFIDF model
    tfidf_model = sklearn.feature_extraction.text.TfidfTransformer()
    tfidf_matrix = tfidf_model.fit_transform(matrix).toarray()

    # Lower dimensionality of original frequency matrix to improve cosine distances for visualization
    reduced_matrix = TruncatedSVD(
        n_components=10, 
        random_state=seed
    ).fit_transform(tfidf_matrix)

    nmf_model = sklearn.decomposition.NMF(
        n_components=num_topics,
        random_state=seed,
        tol=1e-9,
        max_iter=500,
        verbose=False)

    # # Train model
    # doc2topic = nmf_model.fit_transform(tfidf_matrix)
    # topic2token = nmf_model.components_

    # topic_norm = np.sum(topic2token, axis=1)
    # topic2token /= topic_norm[:,np.newaxis]
    # doc2topic *= topic_norm[np.newaxis,:]

    # doc_norm = np.sum(doc2topic, axis=1)
    # doc2topic /= doc_norm[:,np.newaxis]

    doc2topic = model.doc2topic
    # doc2token = model.doc2token
    topic2token = model.topic2token

    # Learn model
    tsne_model = sklearn.manifold.TSNE(
        verbose=True,
        metric='cosine',
        random_state=vis_seed,
        perplexity=20)
    pos = tsne_model.fit_transform(reduced_matrix)

    # Rotate visualization
    theta = np.deg2rad(vis_angle + 60)
    R = np.array([[np.cos(theta), np.sin(theta)], 
                  [-np.sin(theta), np.cos(theta)]])
    pos = np.dot(pos, R)

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
    # print(doc2topic)
    for i in np.random.permutation(len(doc2topic)):
        topic_id = np.argmax(doc2topic[i])
        draw_dot(pos[i], topic_id, zorder)
        zorder += 2

    # Draw legend
    for i in range(num_topics):    
        y = 0.985 - i * 0.02
        label = ', '.join(dic[w] for w in np.argsort(topic2token[i])[::-1][:3])

        draw_dot([0.015, y], i)
        plt.text(0.03, y, label, ha='left', va='center', fontsize=8, zorder=zorder)
        zorder += 1

    # plt.show()

def draw_dot_gensim(model, p, t, zorder=0):
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

def plot_topic_distribution_gensim(model, dic, freqs, fig=None):
    vis_seed = 6 # seed for t-SNE visualization
    vis_angle = 135 # rotation angle for visualization

    gs_tfidf = gensim.models.TfidfModel(freqs)
    corpus_tfidf = gs_tfidf[freqs]
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dic, num_topics=10)
    corpus_lsi = lsi[corpus_tfidf]

    # model = train_nmf_model(dic, corpus_lsi, model.num_topics)

    print("Corpus LSI:", corpus_lsi, corpus_lsi[0])

    reduced_tfidfs = []
    for doc in corpus_lsi:
        reduced_tfidfs.append([score for topic, score in doc])


    # Learn model
    tsne_model = sklearn.manifold.TSNE(
        verbose=True,
        metric='cosine',
        random_state=vis_seed,
        perplexity=20)
    pos = tsne_model.fit_transform(reduced_tfidfs)

    # # Rotate visualization
    # theta = np.deg2rad(vis_angle + 60)
    # R = np.array([[np.cos(theta), np.sin(theta)], 
    #               [-np.sin(theta), np.cos(theta)]])
    # pos = np.dot(pos, R)

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

    # print(model.doc2topic)
    # Draw dots
    for i in np.random.permutation(len(model.doc2topic)):
        topic_id = np.argmax(model.doc2topic[i])
        draw_dot_gensim(model, pos[i], topic_id, zorder)
        zorder += 2

    # Draw legend
    for i in range(model.num_topics):    
        y = 0.985 - i * 0.02
        label = ', '.join(dic[w] for w in np.argsort(model.topic2token[i])[::-1][:3])

        draw_dot_gensim(model, [0.015, y], i)
        plt.text(0.03, y, label, ha='left', va='center', fontsize=8, zorder=zorder)
        zorder += 1

def plot_topic_clouds(model, cols=3, fig=None):
    if fig is None:
        fig = prepare_fig(2)#plt.gcf()
        ax = plt.gca()
    
    rows = int(model.num_topics / float(cols) + cols - 1)

    for i in range(model.num_topics):
        ax = fig.add_subplot(rows, cols, i + 1)
        plot_topic_cloud(model, i, ax=ax)


def plot_topic_cloud(model, topicid, ax=None):
    if ax is None: ax = plt.gca()

    im = generate_topic_cloud(model, topicid)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(generate_topic_cloud(model, topicid), interpolation='bilinear')

def generate_topic_cloud(model, topicid):
    cmap = plt.get_cmap('Blues')
    max_font_size = 75

    mapping = dict()
    maximum = np.amax(model.topic2token[topicid])

    for i in np.argsort(model.topic2token[topicid])[-100:]:
        if model.topic2token[topicid, i] > 0:
            mapping[model.dic[i]] = model.topic2token[topicid, i] / maximum

    def get_color(word, **kwargs):
        weight = kwargs['font_size'] / 75.0 * 0.7 + 0.3
        r, g, b = np.array(cmap(weight)[:3]) * 255
        return 'rgb({}, {}, {})'.format(int(r), int(g), int(b))

    wc = wordcloud.WordCloud(
            prefer_horizontal=True,
            max_font_size=max_font_size,
            background_color='white',
            color_func=get_color,
            scale=2,
            relative_scaling=0.5)
    wc.fit_words(mapping)

    return wc.to_array()


class TopicModel:
    def __init__(self, dic, doc2topic, topic2token):
        self.dic = dic
        self.doc2topic = doc2topic
        self.topic2token = topic2token
        self.num_topics = len(topic2token)
        self.num_documents = len(doc2topic)
        self.num_tokens = len(topic2token.T)

def train_nmf_model(dic, freqs, num_topics=9, **kwargs):
    tfidf_model = gensim.models.TfidfModel(dictionary=dic)
    tmp = [tfidf_model[w] for w in freqs]
    nmf_model = gensim.models.nmf.Nmf(tmp, num_topics, dic, **kwargs)
    topic2token = nmf_model.get_topics()
    doc2topic = np.zeros((len(freqs), num_topics))

    for i in range(len(freqs)):
        for j, f in nmf_model.get_document_topics(freqs[i]):
            doc2topic[i, j] = f

    return TopicModel(dic, doc2topic, topic2token)

def train_lda_model(dic, freqs, num_topics=9, **kwargs):
    model = gensim.models.LdaModel(freqs, num_topics, id2word=dic, **kwargs)
    topic2token = model.get_topics()
    doc2topic = np.zeros((len(freqs), num_topics))

    for i in range(len(freqs)):
        for j, f in model.get_document_topics(freqs[i]):
            doc2topic[i, j] = f
            
    return TopicModel(dic, doc2topic, topic2token)


def build_corpus_simple(docs, stopwords=[], bigrams={}, min_length=2):
    filters = [
        lambda s: [x.lower() for x in s],
        strip_punctuation,
        strip_non_alphanum,
        partial(merge_bigrams, bigrams=bigrams),
        strip_default_stopwords,
        partial(strip_stopwords, stopwords=set(stopwords)),
        partial(strip_short, min_length=min_length),
        stem_smart,
    ]

    return build_corpus(docs, filters)

def build_corpus(docs, filters):
    texts = [(doc.title or ' ') + ' ' + (doc.abstract or ' ') for doc in docs]

    for f in filters:
        texts = f(texts)

    corpus = [text.split(' ') for text in texts]
    dic = gensim.corpora.Dictionary(corpus)
    freqs = [dic.doc2bow(x) for x in corpus]

    return dic, freqs

def merge_bigrams(texts, bigrams):
    # TODO
    return texts

def strip_punctuation(texts):
    return map(gensim.parsing.preprocessing.strip_punctuation, texts)

def strip_short(texts, min_length=3):
    return map(lambda x: gensim.parsing.preprocessing.strip_short(x, min_length), texts)

def strip_numeric(texts):
    return map(gensim.parsing.preprocessing.strip_numeric, texts)

def strip_non_alphanum(texts):
    return map(gensim.parsing.preprocessing.strip_non_alphanum, texts)

def strip_stopwords(texts, stopwords):
    for text in texts:
        yield ' '.join(token for token in text.split() if token not in stopwords)

def strip_default_stopwords(texts):
    return strip_stopwords(texts, gensim.parsing.preprocessing.STOPWORDS)

def stem_porter(texts):
    stemmer = gensim.parsing.PorterStemmer()
    for text in texts:
        yield ' '.join(stemmer.stem(token) for token in text.split())

def stem_smart(texts):
    texts = list(texts)
    stemmer = gensim.parsing.PorterStemmer()
    word_count = defaultdict(int)
    stemming = dict()
    unstemming = dict()

    for text in texts:
        for token in text.split():
            word_count[token] += 1

    for token, _ in sorted(word_count.items(), key=lambda p: p[1]):
        stem = stemmer.stem(token)
        stemming[token] = stem
        unstemming[stem] = token

    return [' '.join(unstemming[stemming[token]] for token in text.split()) for text in texts]
        
