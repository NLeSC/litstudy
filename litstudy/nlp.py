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

def train_nmf_model(dic, freqs, num_topics, seed=0, max_iter=500, **kwargs):
    tfidf_matrix = create_tfidf(freqs, dic)

    # Train NMF model
    nmf_model = sklearn.decomposition.NMF(
	n_components=num_topics,
	random_state=seed,
	tol=1e-9,
	max_iter=max_iter,
    # verbose=True,
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
    for text in texts:
        skip_next = True
        new_text = list(text)
        index = 0

        while index + 1 < len(new_text):
            a, b = new_text[index], new_text[index + 1]
            if (a, b) in bigrams:
                new_text[index:index + 2] = bigrams
            else:
                index += 1

        yield new_text

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
        
