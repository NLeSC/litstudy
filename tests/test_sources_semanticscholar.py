from litstudy.sources.semanticscholar import search_semanticscholar
import os

def test_load_s2_file():
    search_semanticscholar('exascale', limit=10)
    #assert 1==2
