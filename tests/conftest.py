import os
from litstudy.sources.scopus_csv import load_scopus_csv


def pytest_generate_tests(metafunc):
    path = os.path.dirname(__file__) + "/resources/scopus.csv"
    docs = load_scopus_csv(path)
    if "doc" in metafunc.fixturenames:
        metafunc.parametrize("doc", docs)
