import numpy as np
import gensim
import gensim.models.nmf
import sys
import sklearn.feature_extraction.text
import wordcloud
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial

def plot_topic_clouds(model, cols=3, fig=None):
    if fig is None: fig = plt.gcf()
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
        
