import os

from litstudy import load_csv
from litstudy.nlp import build_corpus, Corpus

def test_build_corpus_should_instantiate_Corpus():
    path = os.path.dirname(__file__) + "/resources/scopus.csv"
    docs = load_csv(path)

    TestCorpus = build_corpus(docs=docs,
                              max_tokens=100)

    assert type(TestCorpus) == Corpus
    assert len(TestCorpus.dictionary.items()) == 100

def test_build_corpus_should_filter_words():
    path = os.path.dirname(__file__) + "/resources/scopus.csv"
    docs = load_csv(path)

    remove_words = ['author', 'published']
    TestCorpusComplete = build_corpus(docs=docs)
    TestCorpusFiltered = build_corpus(docs=docs, remove_words=remove_words)

    assert len(TestCorpusComplete.dictionary.items()) > 0
    assert remove_words == [item[1] for item in TestCorpusComplete.dictionary.items() if item[1] in remove_words]
    assert [] == [item for item in TestCorpusFiltered.dictionary.items() if item[1] in remove_words]


